import json
import logging

import pytest

from yamabiko.gazetteer.anilist import AniListUnavailable, Character
from yamabiko.gazetteer.gazetteer import (
    SeriesGazetteer,
    active_series_path,
    build_series_gazetteer,
    cache_needs_rebuild,
    feature_enabled,
    gazetteer_path,
    load_active_series,
    save_active_series,
)
from yamabiko.gazetteer.series_profile import load_profile


# ---------------------------------------------------------------------------
# feature_enabled: the single effective on-switch every entry point gates on.
# True iff BOTH gazetteer.enabled AND romanization.enabled -- a series hint can
# only ever reach the model through RomanizationPipeline, so gazetteer-on with
# romanization-off is UI that silently does nothing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gaz,rom,expected",
    [
        (True, True, True),
        (True, False, False),   # hint has no pipeline to flow through
        (False, True, False),
        (False, False, False),
    ],
)
def test_feature_enabled_requires_both_flags(gaz, rom, expected):
    """feature_enabled is true only when gazetteer.enabled AND
    romanization.enabled are both true.

    Seeded defect: change the helper to `return bool(gaz.get("enabled"))`
    (drop the romanization coupling) -> the (True, False) row expects False
    but gets True -> red.
    """
    cfg = {"gazetteer": {"enabled": gaz}, "romanization": {"enabled": rom}}
    assert feature_enabled(cfg) is expected


def test_feature_enabled_warns_on_flag_divergence(caplog):
    """When gazetteer is on but romanization is off, feature_enabled returns
    False AND logs a WARNING naming the mismatch (so the silent-no-op config
    is visible in the log, not a mystery).

    Seeded defect: delete the `_log.warning(...)` in the divergence branch ->
    no record mentioning 'romanization' -> red.
    """
    cfg = {"gazetteer": {"enabled": True}, "romanization": {"enabled": False}}
    with caplog.at_level(logging.WARNING):
        assert feature_enabled(cfg) is False
    assert any("romanization" in r.message for r in caplog.records), (
        "flag divergence (gazetteer on, romanization off) was not warned about"
    )

def test_feature_enabled_missing_keys_fall_back_to_the_shipped_defaults():
    """A config with the enabled keys DROPPED entirely (validator bug,
    hand-edited override) must behave like the shipped defaults --
    default_config.yaml ships BOTH gazetteer.enabled and
    romanization.enabled true, so the helper's fallbacks must agree: a
    dropped key fails loud (feature visibly on) rather than silently
    disabling the feature. The yaml is loaded here so a future default
    flip forces this pin to be revisited alongside the fallbacks.

    Seeded defect: revert either fallback to ``.get("enabled", False)``
    -> {} reads as disabled -> red. (This IS the pre-fix code; confirmed
    red against it.)
    """
    from pathlib import Path

    import yaml

    import yamabiko

    defaults = yaml.safe_load(
        (Path(yamabiko.__file__).parent / "default_config.yaml")
        .read_text(encoding="utf-8")
    )
    assert feature_enabled(defaults) is True, (
        "shipped defaults changed; re-align feature_enabled's fallbacks"
    )
    assert feature_enabled({}) is True
    assert feature_enabled({"gazetteer": {}, "romanization": {}}) is True


def _g():
    return SeriesGazetteer.from_characters([
        Character("リムル・テンペスト", "Rimuru Tempest", ["Satoru Mikami"]),
        Character("ベニマル", "Benimaru", []),
    ])

def test_lookup_full_native_and_split_parts():
    """The full native and each ・-split part (position-aligned to the EN tokens)
    resolve; a single-token name resolves directly.

    Seeded defect: index only the full native -> リムル lookup returns None, red.
    """
    g = _g()
    assert g.lookup("リムル・テンペスト") == "Rimuru Tempest"
    assert g.lookup("リムル") == "Rimuru"
    assert g.lookup("テンペスト") == "Tempest"
    assert g.lookup("ベニマル") == "Benimaru"

def test_lookup_unknown_returns_none():
    """An unrelated surface is not in the map (so the pipeline hints nothing).

    Seeded defect: return a default EN for misses -> リンゴ lookup non-None, red.
    """
    assert _g().lookup("リンゴ") is None

def test_split_count_mismatch_falls_back_to_full_en():
    """When ・-parts and EN tokens don't align 1:1, each part maps to the FULL
    EN (never a wrong token).

    Seeded defect: zip() truncates silently to the shorter -> the 2nd part maps
    to '' -> assert red.
    """
    g = SeriesGazetteer.from_characters(
        [Character("エレン・イェーガー・ファウンダー", "Eren Yeager", [])])
    assert g.lookup("エレン") == "Eren Yeager"
    assert g.lookup("イェーガー") == "Eren Yeager"

def test_lookup_resolves_katakana_alternative_to_full_en():
    """A katakana entry in Character.alternative indexes to that character's
    full EN, independent of (and not colliding with) any native ・-split key.

    Seeded defect: removing the `for alt in c.alternative` indexing branch in
    from_characters -> lookup on the katakana alias returns None, red.
    """
    g = SeriesGazetteer.from_characters(
        [Character("女神", "Goddess Artemis", ["アルテミス"])])
    assert g.lookup("アルテミス") == "Goddess Artemis"

def test_from_characters_empty_is_empty():
    """from_characters([]) with no characters yields an empty gazetteer: no
    default/seed entries leak into the map.

    Seeded defect: seeding the map with any default entry (e.g. m["x"] = "y")
    before the loop -> g.map == {} is false, red.
    """
    g = SeriesGazetteer.from_characters([])
    assert g.lookup("anything") is None
    assert g.map == {}

def test_gazetteer_round_trips_to_disk(tmp_path):
    """save then load reproduces the map; a missing file loads to None.

    Seeded defect: save without the map key (empty json) -> reload lookup None, red.
    """
    g = _g(); p = tmp_path / "9756.json"; g.save(p)
    g2 = SeriesGazetteer.load(p)
    assert g2 is not None and g2.lookup("リムル") == "Rimuru"
    assert SeriesGazetteer.load(tmp_path / "nope.json") is None

def test_gazetteer_load_returns_none_for_corrupt_json(tmp_path):
    """A file that exists but is not valid JSON loads to None instead of
    raising -- callers treat 'no gazetteer' and 'broken gazetteer' the same
    (soft-hint miss), never crash the pipeline.

    Seeded defect: drop the try/except around json.load in load() -> this
    test raises json.JSONDecodeError instead of asserting None, red.
    """
    p = tmp_path / "broken.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert SeriesGazetteer.load(p) is None

def test_gazetteer_path_builds_under_app_data_gazetteers_subdir(tmp_path, monkeypatch):
    """gazetteer_path(media_id) resolves via paths.app_data_dir(), not cwd or
    a hardcoded string -- a frozen-bundle trap (CLAUDE.md hard rule #8).

    Seeded defect: hardcode gazetteer_path to a literal path (or build it
    from Path(".")) instead of paths.app_data_dir() -> this test's
    YAMABIKO_APPDATA override has no effect on the result, red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    assert gazetteer_path(9756) == tmp_path / "gazetteers" / "9756.json"


# ---------------------------------------------------------------------------
# active_series.json pointer round-trip (save_active_series / load_active_series)
# ---------------------------------------------------------------------------


def test_active_series_round_trips_media_id_and_label_including_non_ascii(
    tmp_path, monkeypatch,
):
    """save_active_series then load_active_series reproduces (media_id,
    label, native) exactly, including non-ASCII text, and leaves no .tmp
    file behind. A pointer written without a native title (one saved before
    the banner carried it) loads with native None.

    Seeded defect: have save_active_series write the wrong shape, drop the
    native field, or have load_active_series return the raw dict instead of
    the tuple -> the equality assert goes red. Drop the os.replace and write
    straight to the real path -> the 'no lingering .tmp' assert still passes
    but the atomicity contract is lost; keeping the tmp-absence assert
    documents it.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    label = "転生したらスライムだった件 (2024)"
    save_active_series(182205, label, native="葬送のフリーレン")

    assert load_active_series() == (182205, label, "葬送のフリーレン")
    # Atomic write must not leave the tmp file lingering.
    assert not active_series_path().with_suffix(".json.tmp").exists()
    assert list(tmp_path.glob("*.tmp")) == [], (
        "save_active_series left a .tmp file behind; os.replace should have "
        "renamed it onto the pointer path"
    )

    save_active_series(182205, label)
    assert load_active_series() == (182205, label, None)


def test_load_active_series_returns_none_when_pointer_absent(tmp_path, monkeypatch):
    """No active_series.json yet (fresh install) -> None, not a crash.

    Seeded defect: drop the OSError catch around read_text -> a missing file
    raises FileNotFoundError instead of returning None, red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    assert load_active_series() is None


@pytest.mark.parametrize(
    "raw, why",
    [
        ("{not valid json", "corrupt JSON"),
        ("[1, 2, 3]", "non-dict JSON (a list)"),
        ('"just a string"', "non-dict JSON (a bare string)"),
        ('{"label": "X"}', "missing media_id"),
        ('{"media_id": "182205", "label": "X"}', "non-int media_id (string)"),
        ('{"media_id": true, "label": "X"}', "boolean media_id (int subclass)"),
        ('{"media_id": 182205, "label": 5}', "non-str label"),
        ('{"media_id": 182205}', "missing label"),
    ],
)
def test_load_active_series_returns_none_for_bad_pointer(
    raw, why, tmp_path, monkeypatch,
):
    """A malformed / mistyped pointer means "no series selected" (None),
    never a raise. Covers every guard in load_active_series.

    Seeded defect: drop the `isinstance(media_id, int)` /
    `not isinstance(media_id, bool)` / `isinstance(label, str)` guards ->
    the boolean-media_id or non-str-label cases return a tuple instead of
    None, red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    active_series_path().parent.mkdir(parents=True, exist_ok=True)
    active_series_path().write_text(raw, encoding="utf-8")
    assert load_active_series() is None, f"expected None for {why}"


# ---------------------------------------------------------------------------
# Orientation-aware native-part <-> EN-token alignment (AniList native is
# surname-first for JA-style names but given-first for Western-style ones;
# a positional zip silently swaps the JA-style case).
# ---------------------------------------------------------------------------


def test_surname_first_native_maps_each_part_to_its_own_token():
    # AniList convention for Japanese-style names: native is surname-first,
    # full is given-first. A positional zip pairs every part with the WRONG
    # token (スバル would hint "Natsuki").
    gaz = SeriesGazetteer.from_characters(
        [Character(native="ナツキ・スバル", full="Subaru Natsuki", alternative=[])]
    )
    assert gaz.lookup("スバル") == "Subaru"
    assert gaz.lookup("ナツキ") == "Natsuki"


def test_given_first_native_keeps_straight_order():
    gaz = SeriesGazetteer.from_characters(
        [Character(native="リムル・テンペスト", full="Rimuru Tempest", alternative=[])]
    )
    assert gaz.lookup("リムル") == "Rimuru"
    assert gaz.lookup("テンペスト") == "Tempest"


def test_unalignable_parts_fall_back_to_full_name():
    # A kanji part cannot be transliterated for scoring; neither orientation
    # is confident, so every part maps to the full English name (never a
    # possibly-swapped single token).
    gaz = SeriesGazetteer.from_characters(
        [Character(native="夜空・メル", full="Mel Yozora", alternative=[])]
    )
    assert gaz.lookup("夜空") == "Mel Yozora"
    assert gaz.lookup("メル") == "Mel Yozora"


# ---------------------------------------------------------------------------
# Atomic disk writes + corrupt-cache quarantine. A truncated cache must be
# MOVED aside (not left in place): a corrupt file reads as "rebuild" to
# cache_needs_rebuild, but it must not linger as the rebuild-failure
# fallback either (a fallback that loads to None is "no hints"). A failed
# pointer write must leave no litter.
# ---------------------------------------------------------------------------


def test_load_quarantines_corrupt_cache_so_a_repick_refetches(tmp_path):
    # A truncated/corrupt cache must be MOVED aside: left in place it would
    # keep resurfacing as tray._GazetteerBuildThread's rebuild-failure
    # fallback, loading to None ("no hints") on every offline re-pick.
    path = tmp_path / "123.json"
    path.write_text("{ truncated", encoding="utf-8")
    assert SeriesGazetteer.load(path) is None
    assert not path.is_file()
    assert path.with_suffix(".json.corrupt.bak").is_file()


def test_save_active_series_failed_write_leaves_no_tmp_litter(tmp_path, monkeypatch):
    monkeypatch.setattr("yamabiko.gazetteer.gazetteer.active_series_path",
                        lambda: tmp_path / "active_series.json")
    monkeypatch.setattr("yamabiko.fsutil.os.replace",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        save_active_series(42, "X")
    leftovers = [p for p in tmp_path.iterdir()]
    assert leftovers == []  # no partial tmp, no truncated pointer


# ---------------------------------------------------------------------------
# Cache schema versioning + pick-time freshness (P1-9). save() writes a
# versioned envelope ({"schema": 2, "built_at": <epoch s>, "map": {...}});
# load() accepts BOTH the envelope and the legacy flat map (the stale cache
# is the offline fallback when a rebuild fails, so it must keep loading);
# cache_needs_rebuild is the single pick-time decision the tray's build
# worker gates the network on. Clocks are injected -- no real sleeps.
# ---------------------------------------------------------------------------


def test_save_writes_schema_2_envelope_with_built_at_epoch(tmp_path):
    """save() persists {"schema": 2, "built_at": <epoch seconds>, "map":
    {...}} so later picks can tell a fresh cache from a stale or
    pre-versioning one (and extractor fixes can bump the schema).

    Seeded defect: revert save() to dumping the bare map -> the schema /
    built_at keys are absent -> red.
    """
    p = tmp_path / "9756.json"
    _g().save(p, now=lambda: 1_700_000_000)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["schema"] == 2
    assert data["built_at"] == 1_700_000_000
    assert data["map"]["ベニマル"] == "Benimaru"


def test_load_reads_legacy_flat_map_as_fallback(tmp_path):
    """A pre-versioning cache (bare {surface: en} dict) still LOADS: the
    stale cache is the offline fallback when a rebuild fails, so refusing
    it would turn 'stale hints' into 'no hints' (worse than the current
    cold-offline behavior). Freshness decisions live in
    cache_needs_rebuild, never in load().

    Seeded defect: make load() require the envelope (return None / raise
    KeyError on a flat map) -> red.
    """
    p = tmp_path / "9756.json"
    p.write_text('{"リムル": "Rimuru"}', encoding="utf-8")
    g = SeriesGazetteer.load(p)
    assert g is not None and g.lookup("リムル") == "Rimuru"
    assert p.is_file(), "a legacy cache must not be quarantined by load()"


def test_load_quarantines_envelope_with_non_dict_map(tmp_path):
    """An envelope whose "map" is not a dict is corrupt: quarantined aside
    (the non-dict-JSON contract) so the next pick refetches, and its
    schema/built_at keys must never masquerade as gazetteer entries.

    Seeded defect: fall through to ``SeriesGazetteer(data)`` for a
    schema-stamped dict without a dict map -> "schema" becomes a lookup
    key -> red.
    """
    p = tmp_path / "9756.json"
    p.write_text('{"schema": 2, "built_at": 1, "map": "oops"}', encoding="utf-8")
    assert SeriesGazetteer.load(p) is None
    assert not p.is_file()
    assert p.with_suffix(".json.corrupt.bak").is_file()


def test_load_refuses_an_envelope_with_a_non_current_schema(tmp_path):
    """An envelope stamped with a schema this binary never wrote (a
    schema-3 cache met after a downgrade) is NOT served: its map
    semantics are unknown here, so serving it 'as fresh' could hint
    wrong. It is quarantined like a non-dict map so the next pick
    refetches; only the exact current schema (and the un-stamped legacy
    flat map) load.

    Seeded defect: serve any envelope with a dict map regardless of the
    schema stamp (the pre-fix code) -> load returns the schema-3 map ->
    red. (Confirmed red against exactly that code.)
    """
    p = tmp_path / "9756.json"
    p.write_text('{"schema": 3, "built_at": 1, "map": {"リムル": "Rimuru"}}',
                 encoding="utf-8")
    assert SeriesGazetteer.load(p) is None
    assert not p.is_file()
    assert p.with_suffix(".json.corrupt.bak").is_file()


@pytest.mark.parametrize(
    "raw, why",
    [
        (None, "missing file"),
        ("{corrupt", "corrupt JSON"),
        ("[1, 2]", "non-dict JSON"),
        ('{"リムル": "Rimuru"}', "legacy pre-schema flat map"),
        # The schema rows carry a fresh PAST stamp (now is pinned to
        # 1_000_000.0): the schema guard alone must flag them; a future
        # stamp would trip the future-built_at clause instead and mask a
        # deleted schema check.
        ('{"schema": 1, "built_at": 999000, "map": {}}',
         "lower schema version"),
        ('{"schema": 3, "built_at": 999000, "map": {}}',
         "newer schema version (downgraded binary)"),
        ('{"schema": true, "built_at": 999000, "map": {}}',
         "boolean schema (int subclass)"),
        ('{"schema": 2, "map": {}}', "missing built_at"),
        ('{"schema": 2, "built_at": "yesterday", "map": {}}',
         "non-numeric built_at"),
    ],
)
def test_cache_needs_rebuild_true_for_unversioned_or_malformed(
    raw, why, tmp_path,
):
    """Every absent / pre-versioning / malformed / wrong-schema cache
    shape reads as 'rebuild on the next pick' -- and the probe never
    touches the file itself (the caller keeps it as the offline
    fallback).

    Seeded defect: revert the pick gate to a bare path.is_file() check ->
    every on-disk shape here reads 'no rebuild needed' -> red. Or relax
    the schema guard back to `< _SCHEMA_VERSION` -> the newer-schema row
    (fresh stamp) reads 'no rebuild needed' -> red.
    """
    p = tmp_path / "9756.json"
    if raw is not None:
        p.write_text(raw, encoding="utf-8")
    assert cache_needs_rebuild(p, now=lambda: 1_000_000.0) is True, why
    if raw is not None:
        assert p.is_file(), (
            "the freshness probe must never move/delete the cache; "
            "fallback loading depends on the file staying put"
        )


def test_cache_needs_rebuild_false_within_24h_then_true_after(tmp_path):
    """The pick-time freshness horizon over a real save(): a just-built
    cache is a plain hit (no network) for 24 h, then rebuilds -- an
    airing series' cast must not stay frozen at first-pick state forever.
    Boundary pinned: exactly 24 h old counts as stale.

    Seeded defect: drop the built_at age comparison (schema check only)
    -> the 24 h / 25 h rows never go stale -> red.
    """
    p = tmp_path / "9756.json"
    built = 1_700_000_000
    _g().save(p, now=lambda: built)
    day = 24 * 60 * 60
    assert cache_needs_rebuild(p, now=lambda: built) is False
    assert cache_needs_rebuild(p, now=lambda: built + day - 1) is False
    assert cache_needs_rebuild(p, now=lambda: built + day) is True
    assert cache_needs_rebuild(p, now=lambda: built + day + 3600) is True


def test_cache_needs_rebuild_true_when_built_at_is_ahead_of_the_clock(tmp_path):
    """A built_at AHEAD of the current clock reads as stale, not fresh: a
    clock that was ahead at save time and corrected afterwards (NTP)
    would otherwise read as fresh until the wall clock catches up --
    freezing an airing series' cast at first-pick state, the exact
    symptom the 24 h horizon exists to prevent. built_at == now stays
    fresh (age 0); one second ahead is stale.

    Seeded defect: drop the future-stamp clause (age check only) -> the
    negative age never reaches 24 h -> False forever -> red. (This IS
    the pre-fix code; confirmed red against it.)
    """
    p = tmp_path / "9756.json"
    built = 1_700_000_000
    _g().save(p, now=lambda: built)
    assert cache_needs_rebuild(p, now=lambda: built - 1) is True
    assert cache_needs_rebuild(p, now=lambda: built - 7 * 24 * 3600) is True
    assert cache_needs_rebuild(p, now=lambda: built) is False


# ---------------------------------------------------------------------------
# BACKGROUND role filter. AniList BACKGROUND entries include generic nouns as
# character "names" (店員 / 職員, live-observed 2026-07-10 on Yani Neko), so an
# entry would hint 店員 -> "Tenin" on ordinary dialogue.
# ---------------------------------------------------------------------------


def test_from_characters_excludes_background_role_characters():
    """A role == "BACKGROUND" character produces NO gazetteer entry (native,
    split parts, or alternatives), while MAIN / SUPPORTING / legacy role-less
    characters still do.

    Seeded defect (the mandated drill): invert the filter (keep ONLY
    BACKGROUND) -> 店員 resolves and 素子 does not -> both halves red.
    """
    g = SeriesGazetteer.from_characters([
        Character("素子", "Motoko Kusanagi", [], role="MAIN"),
        Character("店員", "Tenin", ["テンイン"], role="BACKGROUND"),
        # 3-arg legacy construction (role defaults None) must stay indexed.
        Character("ベニマル", "Benimaru", []),
    ])
    assert g.lookup("店員") is None
    assert g.lookup("テンイン") is None
    assert "Tenin" not in g.map.values()
    assert g.lookup("素子") == "Motoko Kusanagi"
    assert g.lookup("ベニマル") == "Benimaru"


# ---------------------------------------------------------------------------
# build_series_gazetteer: the full per-pick build -- picked cast + relations
# carry (related-season casts share speakers) + the series-profile hook, all
# from ONE intel request (the ~13-request pick budget, 30 req/min limit).
# ---------------------------------------------------------------------------


def _rel(relation_type: str, media_id: int, media_type: str = "ANIME") -> dict:
    # Shape of one fetch_media_intel()["relations"] row.
    return {"relationType": relation_type, "id": media_id, "type": media_type,
            "format": "TV", "title": f"Related {media_id}"}


class _FakeBuildClient:
    """AniListClient stand-in for build_series_gazetteer (the wire client
    has its own mocked-httpx + real-tier suites); records every character
    fetch (media_id, max_pages) so the per-pick request budget is
    assertable, can fail a chosen related media's fetch, and reports the
    page shares ``pages`` gives a media through on_page (default: one
    page, 1.0)."""

    def __init__(self, relations=None, rosters=None, fail_ids=(), pages=None):
        self._relations = relations or []
        self._rosters = rosters or {}
        self._fail_ids = set(fail_ids)
        self._pages = pages or {}
        self.intel_calls: list[int] = []
        self.char_calls: list[tuple[int, int]] = []

    def fetch_media_intel(self, media_id: int) -> dict:
        self.intel_calls.append(media_id)
        return {"genres": [], "tags": [], "description": "Yani is a catgirl.",
                "relations": list(self._relations)}

    def fetch_characters(self, media_id: int, max_pages: int = 6, on_page=None):
        self.char_calls.append((media_id, max_pages))
        if media_id in self._fail_ids:
            raise AniListUnavailable("simulated related-media failure")
        if on_page is not None:
            for share in self._pages.get(media_id, (1.0,)):
                on_page(share)
        return list(self._rosters.get(media_id, []))


def test_build_series_gazetteer_merges_a_related_anime_cast(tmp_path):
    """A related ANIME (SIDE_STORY) cast merges into the picked media's
    gazetteer: names that only exist in the related season resolve too.

    Seeded defect: skip the carry loop (build from picked chars only) ->
    イモコ resolves to None -> red.
    """
    client = _FakeBuildClient(
        relations=[_rel("SIDE_STORY", 208105)],
        rosters={
            207141: [Character("ヤニ", "Yani", [])],
            208105: [Character("イモコ", "Imoko Satou", [])],
        },
    )
    g = build_series_gazetteer(client, 207141, tmp_path)
    assert g.lookup("ヤニ") == "Yani"
    assert g.lookup("イモコ") == "Imoko Satou"


def test_build_series_gazetteer_collision_keeps_picked_media_entry(tmp_path):
    """A related character whose native name collides with a picked-media
    character is dropped WHOLE: the picked spelling wins, and the related
    duplicate cannot smuggle entries in via its alternative aliases.

    Seeded defect: drop the seen-native dedupe (append every related
    char) -> the related duplicate's ヤニコ alias maps to 'Cigarette Cat'
    -> red. (The map-level setdefault alone would hide that defect for
    the native key, so the alias assert is the load-bearing one.)
    """
    client = _FakeBuildClient(
        relations=[_rel("SEQUEL", 208105)],
        rosters={
            207141: [Character("ヤニ", "Yani", [])],
            208105: [Character("ヤニ", "Cigarette Cat", ["ヤニコ"])],
        },
    )
    g = build_series_gazetteer(client, 207141, tmp_path)
    assert g.lookup("ヤニ") == "Yani"
    assert g.lookup("ヤニコ") is None
    assert "Cigarette Cat" not in g.map.values()


def test_build_series_gazetteer_entryless_picked_chars_do_not_block_related_entries(
    tmp_path,
):
    """A picked-media character that from_characters will FILTER OUT (a
    BACKGROUND extra, or one with no English full name) claims no
    dedup slot in the relations carry: a same-native MAIN / SUPPORTING
    character from a related media still gets its entry, instead of
    both being silently lost.

    Seeded defect: seed `seen` from ALL picked characters (drop the
    entry-eligibility guard) -> 店員 and カヲル both resolve to None -> red.
    """
    client = _FakeBuildClient(
        relations=[_rel("SEQUEL", 999)],
        rosters={
            207141: [
                Character("ヤニ", "Yani", []),
                Character("店員", "Tenin", [], role="BACKGROUND"),
                Character("カヲル", None, []),   # no EN name -> never an entry
            ],
            999: [
                Character("店員", "Clerk Boss", [], role="MAIN"),
                Character("カヲル", "Kaworu", [], role="SUPPORTING"),
            ],
        },
    )
    g = build_series_gazetteer(client, 207141, tmp_path)
    assert g.lookup("店員") == "Clerk Boss"   # related MAIN takes the slot
    assert g.lookup("カヲル") == "Kaworu"
    assert "Tenin" not in g.map.values()      # BACKGROUND extra still filtered
    assert g.lookup("ヤニ") == "Yani"         # picked cast untouched


def test_build_series_gazetteer_related_background_extra_does_not_block_a_later_relation(
    tmp_path,
):
    """A BACKGROUND character in the FIRST related media neither becomes
    an entry nor claims the dedup slot: the SECOND related media's
    same-native MAIN character still merges. (Same silent-loss hazard as
    the picked-media case, one relation deeper.)

    Seeded defect: append filtered-out related chars into the merge (drop
    the loop's eligibility skip) -> the first relation's BACKGROUND
    マスター claims the slot -> マスター resolves to None -> red.
    """
    client = _FakeBuildClient(
        relations=[_rel("PREQUEL", 10), _rel("SEQUEL", 20)],
        rosters={
            207141: [Character("ヤニ", "Yani", [])],
            10: [Character("マスター", "Master", [], role="BACKGROUND")],
            20: [Character("マスター", "Master Kouji", [], role="MAIN")],
        },
    )
    g = build_series_gazetteer(client, 207141, tmp_path)
    assert g.lookup("マスター") == "Master Kouji"
    assert "Master" not in g.map.values()


def test_build_series_gazetteer_carries_two_anime_relations_in_priority_order(
    tmp_path,
):
    """The carry fetches at most 2 related media, ANIME only, ranked
    PREQUEL > SEQUEL > PARENT > SIDE_STORY, at 2 pages each -- one intel
    request total. A MANGA PREQUEL (top priority, wrong medium) is
    ignored; PARENT / SIDE_STORY anime lose to PREQUEL / SEQUEL; the
    picked fetch keeps its own (default) page cap.

    Seeded defect: lift the 2-media cap -> media 30 fetched -> red; or
    rank by list order instead of relation type -> 40 fetched before
    10 -> red; or fetch MANGA relations -> 50 fetched -> red; or pass
    the related fetches max_pages=6 -> the (10, 2)/(20, 2) rows red;
    or re-fetch intel per relation -> intel_calls red.
    """
    client = _FakeBuildClient(
        relations=[
            _rel("SIDE_STORY", 40),
            _rel("PREQUEL", 50, media_type="MANGA"),
            _rel("SEQUEL", 20),
            _rel("PREQUEL", 10),
            _rel("PARENT", 30),
        ],
        rosters={
            207141: [Character("ヤニ", "Yani", [])],
            10: [Character("レイ", "Rei", [])],
            20: [Character("アスカ", "Asuka", [])],
            30: [Character("ミサト", "Misato", [])],
            40: [Character("カヲル", "Kaworu", [])],
            50: [Character("ゲンドウ", "Gendou", [])],
        },
    )
    g = build_series_gazetteer(client, 207141, tmp_path)
    assert client.intel_calls == [207141]
    assert client.char_calls == [(207141, 6), (10, 2), (20, 2)]
    assert g.lookup("レイ") == "Rei"
    assert g.lookup("アスカ") == "Asuka"
    assert g.lookup("ミサト") is None
    assert g.lookup("カヲル") is None
    assert g.lookup("ゲンドウ") is None


def test_build_series_gazetteer_survives_a_related_fetch_failure(tmp_path):
    """One related media's character fetch failing (rate limit / outage
    mid-pick) must not throw away the already-fetched picked cast: the
    failed relation is skipped (logged) and the remaining one still
    merges.

    Seeded defect: remove the try/except around the related fetch ->
    AniListUnavailable escapes and no gazetteer is returned -> red.
    """
    client = _FakeBuildClient(
        relations=[_rel("PREQUEL", 10), _rel("SEQUEL", 20)],
        rosters={
            207141: [Character("ヤニ", "Yani", [])],
            20: [Character("アスカ", "Asuka", [])],
        },
        fail_ids={10},
    )
    g = build_series_gazetteer(client, 207141, tmp_path)
    assert g.lookup("ヤニ") == "Yani"        # picked cast survives
    assert g.lookup("アスカ") == "Asuka"     # later relation still carried
    assert (10, 2) in client.char_calls      # the failed one WAS attempted


def test_build_series_gazetteer_reports_staged_monotonic_progress(tmp_path):
    """The pick's progress line follows the real requests (phase-5 L10):
    0.15 once the intel request is done, 0.15 + 0.6 x the cast share per
    picked page (0.75 on the last), then 0.75 + 0.25 x (i + 1) / n after
    each related fetch, the failed one included.

    Seeded defects: no report after the intel step -> the first value red;
    report the raw page share -> 0.5 / 1.0 red; skip the report for a failed
    related fetch -> four values -> red.
    """
    client = _FakeBuildClient(
        relations=[_rel("PREQUEL", 10), _rel("SEQUEL", 20)],
        rosters={
            207141: [Character("ヤニ", "Yani", [])],
            20: [Character("アスカ", "Asuka", [])],
        },
        fail_ids=(10,),
        pages={207141: (0.5, 1.0)},
    )
    reported: list[float] = []
    build_series_gazetteer(client, 207141, tmp_path, on_progress=reported.append)
    assert reported == pytest.approx([0.15, 0.45, 0.75, 0.875, 1.0])


def test_build_series_gazetteer_persists_the_series_profile_for_the_pick(
    tmp_path,
):
    """The pick's single intel request also materializes the series
    profile on disk: load_profile finds it beside the gazetteer cache
    afterwards (Task C reads it for the context hint without refetching).

    Seeded defect: drop the build_profile call from
    build_series_gazetteer -> load_profile returns None -> red.
    """
    client = _FakeBuildClient(rosters={207141: [Character("ヤニ", "Yani", [])]})
    build_series_gazetteer(client, 207141, tmp_path)
    profile = load_profile(207141, tmp_path)
    assert profile is not None
    assert profile.media_id == 207141
    assert profile.synopsis_first_sentence == "Yani is a catgirl."


def test_build_series_gazetteer_survives_a_profile_fetch_failure(tmp_path, caplog):
    """The series profile only adds related seasons and a context hint; a
    failed intel request (rate limit / outage) must not cost the user the
    picked cast. The pick degrades to the picked media's characters.

    Seeded defect: remove the try/except around build_profile ->
    AniListUnavailable escapes and no gazetteer is returned -> red.
    """

    class _IntelDown(_FakeBuildClient):
        def fetch_media_intel(self, media_id: int) -> dict:
            self.intel_calls.append(media_id)
            raise AniListUnavailable("simulated intel failure")

    client = _IntelDown(rosters={207141: [Character("ヤニ", "Yani", [])]})
    with caplog.at_level(logging.WARNING, logger="yamabiko.gazetteer.gazetteer"):
        g = build_series_gazetteer(client, 207141, tmp_path)
    assert g.lookup("ヤニ") == "Yani"
    assert client.char_calls == [(207141, 6)]   # no relations to carry
    assert "series profile fetch failed" in caplog.text
