"""Tests for GlossaryManager: lookup priority, roundtrip, genre switching."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from yamabiko.romanization.glossary import GlossaryManager


@pytest.fixture
def tmp_glossaries(tmp_path: Path) -> Path:
    """Create a temporary glossaries dir with 4 empty genre files + empty user.yaml."""
    gdir = tmp_path / "glossaries"
    gdir.mkdir()
    for genre in ("variety", "drama", "news", "anime"):
        (gdir / f"{genre}.json").write_text(
            json.dumps({"version": 1, "entries": {}}), encoding="utf-8"
        )
    (gdir / "user.yaml").write_text(
        yaml.safe_dump({"entries": {}}, allow_unicode=True), encoding="utf-8"
    )
    return gdir


def _write_genre(gdir: Path, genre: str, entries: dict) -> None:
    """Seed a genre JSON on disk in the manager's own file format.

    The learning loop that once populated the genre layer at runtime was
    removed, so tests seed the layer the way an upgraded install carries it:
    by writing the ``{genre}.json`` the manager loads on construction.
    """
    (gdir / f"{genre}.json").write_text(
        json.dumps({"version": 1, "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_empty_manager_returns_none_for_unknown(tmp_glossaries: Path):
    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    assert mgr.lookup("田中") is None


def test_genre_entry_lookup_roundtrip(tmp_glossaries: Path):
    """An entry in the genre JSON is returned by lookup with both its English
    form and its furigana reading.

    Seeded defect: drop the `if jp in self._entries` branch from lookup ->
    田中 is no longer found in the genre layer -> red.
    """
    _write_genre(tmp_glossaries, "variety", {
        "田中": {"en": "Tanaka", "reading": "たなか"},
    })
    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    entry = mgr.lookup("田中")
    assert entry is not None
    assert entry.en == "Tanaka"
    assert entry.reading == "たなか"


def test_save_fsyncs_flushed_temp_file_before_replace(
    tmp_glossaries: Path, monkeypatch,
):
    """os.replace journals the NTFS RENAME atomically but does NOT flush
    the temp file's data blocks. Without flush+fsync before the
    replace, power loss shortly after a save publishes a zero-length /
    partial genre JSON; on next launch _load_genre_unlocked quarantines
    it and every entry in that genre silently vanishes.

    Asserts the established fsync-before-replace idiom (mirrors
    tests/test_config_writer.py::
    test_atomic_write_fsyncs_flushed_temp_file_before_replace): at
    least one os.fsync runs BEFORE os.replace, and at fsync time the
    fd already holds data (size > 0 -- a single buffered fh.write
    stays in the Python userspace buffer until flushed, so the size
    clause makes the flush load-bearing for the json path).

    Seeded defect: delete ``fh.flush(); os.fsync(fh.fileno())`` from
    _save_unlocked -> no fsync event before the replace -> red.
    """
    import os

    _write_genre(tmp_glossaries, "variety", {"東京": {"en": "Tokyo"}})

    events: list[tuple[str, int]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_fsync(fd):
        events.append(("fsync", os.fstat(fd).st_size))
        return real_fsync(fd)

    def recording_replace(src, dst):
        events.append(("replace", -1))
        return real_replace(src, dst)

    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)

    mgr.save()

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
        f"no os.fsync of flushed data before os.replace -- the genre-"
        f"glossary write is not durable across power loss. "
        f"Events: {events!r}"
    )


def test_manager_init_survives_non_utf8_user_yaml(tmp_glossaries: Path):
    # A mis-encoded hand-edited user.yaml must degrade to "user layer
    # empty", not raise out of __init__ (which main.py's catch-all turns
    # into romanization disabled entirely).
    # Seeded defect: let UnicodeDecodeError escape load_yaml_glossary.
    (tmp_glossaries / "user.yaml").write_bytes(
        "entries:\n  田中:\n    en: Tanaka\n".encode("cp932")
    )
    (tmp_glossaries / "variety.json").write_text(
        json.dumps({"version": 1, "entries": {
            "東京": {"en": "Tōkyō", "reading": "とうきょう"}
        }}),
        encoding="utf-8",
    )
    mgr = GlossaryManager(tmp_glossaries, genre="variety")  # must not raise
    entry = mgr.lookup("東京")
    assert entry is not None and entry.en == "Tōkyō"


def test_user_yaml_overrides_genre_json(tmp_glossaries: Path):
    (tmp_glossaries / "variety.json").write_text(
        json.dumps({"version": 1, "entries": {
            "田中": {"en": "Tanaka", "reading": "たなか"}
        }}),
        encoding="utf-8",
    )
    (tmp_glossaries / "user.yaml").write_text(
        yaml.safe_dump({"entries": {
            "田中": {"en": "Mr. Tanaka", "reading": "たなか"}
        }}, allow_unicode=True),
        encoding="utf-8",
    )
    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    entry = mgr.lookup("田中")
    assert entry.en == "Mr. Tanaka"


def test_genre_switch_loads_different_entries(tmp_glossaries: Path):
    (tmp_glossaries / "variety.json").write_text(
        json.dumps({"version": 1, "entries": {
            "田中": {"en": "Tanaka", "reading": "たなか"}
        }}),
        encoding="utf-8",
    )
    (tmp_glossaries / "drama.json").write_text(
        json.dumps({"version": 1, "entries": {
            "山田": {"en": "Yamada", "reading": "やまだ"}
        }}),
        encoding="utf-8",
    )
    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    assert mgr.lookup("田中") is not None
    assert mgr.lookup("山田") is None

    mgr.set_genre("drama")
    assert mgr.lookup("田中") is None
    assert mgr.lookup("山田") is not None


def test_invalid_genre_raises(tmp_glossaries: Path):
    with pytest.raises(ValueError):
        GlossaryManager(tmp_glossaries, genre="sports")


def test_save_round_trips_loaded_genre_entries(tmp_glossaries: Path):
    """Loading a genre JSON and re-saving it preserves every entry field and
    does not duplicate on a repeat save (a save must serialize what it
    loaded, not drop the reading/count or grow the set).

    Seeded defect: have _save_unlocked emit `asdict(e)` minus a field (e.g.
    drop 'reading') -> the round-tripped 田中 loses たなか -> red.
    """
    _write_genre(tmp_glossaries, "variety", {
        "田中": {
            "en": "Tanaka",
            "reading": "たなか",
            "count": 3,
            "first_seen": "2026-01-01T00:00:00+00:00",
        },
    })
    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    mgr.save()
    mgr.save()
    data = json.loads((tmp_glossaries / "variety.json").read_text(encoding="utf-8"))
    assert list(data["entries"]) == ["田中"]
    entry = data["entries"]["田中"]
    assert entry["en"] == "Tanaka"
    assert entry["reading"] == "たなか"
    assert entry["count"] == 3


def test_seed_reading_outranks_legacy_genre_entry(tmp_glossaries: Path, tmp_path: Path):
    """user.yaml > _seed.yaml > genre.json for the same surface: the curated
    seed reading にほん must win over a legacy genre entry's にっぽん.

    Seeded defect: check the genre layer before the seed in lookup() -> the
    genre entry's にっぽん is returned -> red."""
    seed_path = tmp_path / "_seed.yaml"
    seed_path.write_text(
        "version: 1\n"
        "entries:\n"
        "  日本:\n"
        "    en: Japan\n"
        "    reading: にほん\n",
        encoding="utf-8",
    )
    _write_genre(tmp_glossaries, "variety", {
        "日本": {"en": "Nippon", "reading": "にっぽん", "count": 3,
                 "first_seen": "2026-01-01T00:00:00+00:00"},
    })
    gm = GlossaryManager(tmp_glossaries, genre="variety", seed_path=seed_path)
    entry = gm.lookup("日本")
    assert entry is not None
    assert entry.reading == "にほん"


def test_seed_fills_missing_entries(tmp_glossaries: Path, tmp_path: Path):
    """Seed covers entries not present in genre or user."""
    seed_path = tmp_path / "_seed.yaml"
    seed_path.write_text(
        "version: 1\n"
        "entries:\n"
        "  大阪府:\n"
        "    en: Ōsaka\n"
        "    reading: おおさかふ\n",
        encoding="utf-8",
    )
    gm = GlossaryManager(tmp_glossaries, genre="variety", seed_path=seed_path)
    entry = gm.lookup("大阪府")
    assert entry is not None
    assert entry.en == "Ōsaka"


def test_user_overrides_seed(tmp_glossaries: Path, tmp_path: Path):
    """user.yaml beats seed for same surface."""
    seed_path = tmp_path / "_seed.yaml"
    seed_path.write_text(
        "version: 1\n"
        "entries:\n"
        "  田中:\n"
        "    en: Tanaka\n"
        "    reading: たなか\n",
        encoding="utf-8",
    )
    (tmp_glossaries / "user.yaml").write_text(
        yaml.safe_dump(
            {"entries": {"田中": {"en": "Mr. Tanaka", "reading": "たなか"}}},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    gm = GlossaryManager(tmp_glossaries, genre="variety", seed_path=seed_path)
    entry = gm.lookup("田中")
    assert entry is not None
    assert entry.en == "Mr. Tanaka"


def test_no_seed_still_works(tmp_glossaries: Path):
    """GlossaryManager works without seed_path."""
    gm = GlossaryManager(tmp_glossaries, genre="variety")
    assert gm.lookup("nonexistent") is None


def test_save_never_writes_the_seed_file(tmp_glossaries: Path, tmp_path: Path):
    """A save flushes only the active genre JSON, never the shipped seed.

    Seeded defect: point _save_unlocked at the seed path (or add a seed
    flush) -> the seed bytes change -> red.
    """
    seed_path = tmp_path / "_seed.yaml"
    seed_text = (
        "version: 1\n"
        "entries:\n"
        "  東京:\n"
        "    en: Tokyo\n"
        "    reading: とうきょう\n"
    )
    seed_path.write_text(seed_text, encoding="utf-8")
    gm = GlossaryManager(tmp_glossaries, genre="variety", seed_path=seed_path)
    gm.save()
    assert seed_path.read_text(encoding="utf-8") == seed_text


# --- corrupt genre JSON must not crash startup ---


def test_corrupt_genre_json_quarantines_and_starts_empty(tmp_glossaries: Path):
    """A truncated/corrupt genre JSON must not crash GlossaryManager.__init__
    (it's read unconditionally); quarantine it and start the genre empty.

    Seeded-defect drill: drop the try/except in _load_genre_unlocked ->
    json.loads raises -> GlossaryManager(...) raises -> this errors.
    """
    (tmp_glossaries / "variety.json").write_text("{ this is not json", encoding="utf-8")
    mgr = GlossaryManager(tmp_glossaries, genre="variety")  # must NOT raise
    assert mgr.lookup("anything") is None
    assert (tmp_glossaries / "variety.json.corrupt.bak").exists(), (
        "corrupt genre JSON was not quarantined to .corrupt.bak"
    )


# --- genre JSON writes must be atomic ---


def test_save_is_atomic_original_survives_failed_replace(
    tmp_glossaries: Path, monkeypatch,
):
    """An interrupted write must not truncate/corrupt the existing glossary.

    Seeded-defect drill: revert _save_unlocked to a plain write_text ->
    the file is overwritten in place (no os.replace to fail), so the save
    both dodges the injected OSError and mutates the file -> both the
    pytest.raises and the 'original survives' assertions fail.
    """
    import os as _os
    _write_genre(tmp_glossaries, "variety", {"東京": {"en": "Tokyo"}})
    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    original = (tmp_glossaries / "variety.json").read_text(encoding="utf-8")

    def _boom(*a, **k):
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(_os, "replace", _boom)
    with pytest.raises(OSError):
        mgr.save()
    assert (tmp_glossaries / "variety.json").read_text(encoding="utf-8") == original, (
        "a failed write corrupted/truncated the existing glossary"
    )
    leftovers = [p.name for p in tmp_glossaries.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"atomic write left temp files behind: {leftovers}"
