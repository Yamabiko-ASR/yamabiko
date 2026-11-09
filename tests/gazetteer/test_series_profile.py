"""Tests for yamabiko.gazetteer.series_profile -- the per-series
intelligence layer (profile ingest, disk cache, context-hint composer).

Unit tier per docs/TESTING.md Sec 4: the AniList wire client is faked
here because these tests claim ingest/cache/composition logic, not the
wire -- the client's own parsing and failure taxonomy are covered in
tests/gazetteer/test_anilist.py (mocked httpx) and the live endpoint in
tests/gazetteer/test_real_anilist.py (hard rule #3). context_hint is a
pure function, so those tests build SeriesProfile values directly.

Fixture shapes are live-observed AniList payloads (probed 2026-07-10):
Yani Neko = media 207141 (description with <BR> + a "(Source: ..."
attribution that the API has served TRUNCATED mid-word, unclosed), and
GITS 2026 = media 177699 (the tag rank/category spread the composer
filters).
"""
from __future__ import annotations

import json
import logging

import pytest

from yamabiko.gazetteer.series_profile import (
    Relation,
    SeriesProfile,
    Tag,
    build_profile,
    context_hint,
    load_profile,
    profile_path,
)


class _FakeClient:
    """Stands in for AniListClient.fetch_media_intel only (the single wire
    call build_profile is allowed); records call ids so tests can assert
    the one-request budget."""

    def __init__(self, intel: dict):
        self._intel = intel
        self.intel_calls: list[int] = []

    def fetch_media_intel(self, media_id: int) -> dict:
        self.intel_calls.append(media_id)
        return self._intel


def _tag_row(name: str, rank: int, category: str, *, media_spoiler=False,
             general_spoiler=False) -> dict:
    return {"name": name, "rank": rank, "category": category,
            "isMediaSpoiler": media_spoiler, "isGeneralSpoiler": general_spoiler}


# Live-observed Yani Neko (207141) description shape: <BR> paragraph
# breaks and the trailing Source attribution served truncated mid-word
# with the paren never closed -- the cleaner must survive exactly this.
_YANI_DESCRIPTION = (
    "Yani is a catgirl with a seriously bad smoking habit. She smokes so "
    "much that her apartment smells like ash and is littered with "
    "cigarette butts—and plenty of other trash! Will she ever get her "
    "life together? \n<BR><BR>\n(Source: Seven Seas Entertainm"
)
_YANI_FIRST_SENTENCE = "Yani is a catgirl with a seriously bad smoking habit."


def _intel(description: str = "", tags: list | None = None,
           genres: list | None = None, relations: list | None = None) -> dict:
    return {"genres": genres or [], "tags": tags or [],
            "description": description, "relations": relations or []}


# ------------------------------------------------------------ ingest: tags

def test_build_profile_drops_spoiler_flagged_tags_at_ingest(tmp_path):
    """Tags with isMediaSpoiler or isGeneralSpoiler true never enter the
    profile, so no downstream consumer can leak one into a prompt.

    Seeded defect: drop the spoiler-flag filter in build_profile -> 'Time
    Manipulation' / 'Conspiracy' appear in profile.tags -> red.
    """
    client = _FakeClient(_intel(tags=[
        _tag_row("Nekomimi", 90, "Cast-Traits"),
        _tag_row("Time Manipulation", 60, "Theme-Sci-Fi",
                 media_spoiler=True, general_spoiler=True),
        _tag_row("Conspiracy", 80, "Theme-Drama", media_spoiler=True),
    ]))
    profile = build_profile(client, 207141, tmp_path)
    assert [t.name for t in profile.tags] == ["Nekomimi"]
    assert profile.tags[0] == Tag("Nekomimi", 90, "Cast-Traits")


# -------------------------------------------------------- ingest: synopsis

def test_build_profile_takes_first_sentence_of_live_yani_neko_description(tmp_path):
    """The live-observed Yani Neko description (<BR> breaks, truncated
    UNCLOSED '(Source: Seven Seas Entertainm' suffix) cleans to exactly
    its first sentence, terminator kept.

    Seeded defect: split at m.start() instead of m.end() (drop the '.')
    -> the terminator missing -> red. Or return the whole cleaned text
    -> the second sentence rides along -> red.
    """
    client = _FakeClient(_intel(description=_YANI_DESCRIPTION))
    profile = build_profile(client, 207141, tmp_path)
    assert profile.synopsis_first_sentence == _YANI_FIRST_SENTENCE


def test_build_profile_strips_html_tags_inside_the_first_sentence(tmp_path):
    """HTML tags BEFORE the first terminator (live TenSura S4 shape:
    'of <i>Title</i>.<br><br>') are stripped, not carried into the hint.

    Seeded defect: drop the HTML-tag strip -> '<i>' survives inside the
    sentence -> red.
    """
    client = _FakeClient(_intel(description=(
        "The fourth season of <i>Tensei Shitara Slime Datta Ken</i>."
        "<br><br>\nDemon Lord Rimuru's dream takes a step closer."
    )))
    profile = build_profile(client, 182205, tmp_path)
    assert profile.synopsis_first_sentence == (
        "The fourth season of Tensei Shitara Slime Datta Ken."
    )


def test_build_profile_strips_markdown_emphasis_markers(tmp_path):
    """Markdown emphasis markers (__bold__ / *italic* / _underscore_) are
    removed while their text is kept.

    Seeded defect: drop the markdown strip -> '__Yani__' survives -> red.
    """
    client = _FakeClient(_intel(
        description="__Yani__ is a *catgirl*, and _nothing_ else. More text."))
    profile = build_profile(client, 207141, tmp_path)
    assert profile.synopsis_first_sentence == (
        "Yani is a catgirl, and nothing else."
    )


def test_build_profile_removes_spoiler_blocks_with_their_text(tmp_path):
    """An AniList ~!spoiler!~ block is removed WITH its enclosed text --
    stripping only the delimiters would expose the spoiler, unmarked, in
    the synopsis (and from there in the translation prompt), defeating
    the same spoiler-safety contract the tag filter enforces.

    Seeded defect: strip `~!` / `!~` as emphasis markers (keep the
    enclosed text) -> 'the mentor is the villain' surfaces -> red.
    """
    client = _FakeClient(_intel(description=(
        "Everything seems fine at first, but ~!the mentor is the villain "
        "all along!~ in this quiet town. Life goes on."
    )))
    profile = build_profile(client, 207141, tmp_path)
    assert "villain" not in profile.synopsis_first_sentence
    assert "~" not in profile.synopsis_first_sentence
    assert profile.synopsis_first_sentence == (
        "Everything seems fine at first, but in this quiet town."
    )


def test_build_profile_drops_an_unclosed_spoiler_block_through_the_end(tmp_path):
    """A spoiler opener with no closer (the live API serves descriptions
    truncated mid-text -- observed on the Source suffix -- so a cut can
    land inside a spoiler) drops through the END of the text: the
    fail-safe direction is over-removal, never exposing a half-spoiler.

    Seeded defect: require the closing !~ (drop the spoiler regex's `|$`
    branch) -> the unclosed spoiler text survives as the synopsis -> red.
    """
    client = _FakeClient(_intel(description=(
        "In a quiet village, ~!the elder is secretly the sea god"
    )))
    profile = build_profile(client, 207141, tmp_path)
    assert profile.synopsis_first_sentence == "In a quiet village,"


def test_build_profile_strips_truncated_source_suffix_when_no_terminator(tmp_path):
    """A description with NO sentence terminator keeps the whole cleaned
    text -- which means the trailing (unclosed, truncated) Source
    attribution must have been stripped, or it leaks into the hint.

    Seeded defect: require a closing ')' in the Source regex -> the
    truncated 'Entertainm' tail survives -> red.
    """
    client = _FakeClient(_intel(description=(
        "A catgirl comedy with no closing punctuation\n<BR>"
        "(Source: Seven Seas Entertainm"
    )))
    profile = build_profile(client, 207141, tmp_path)
    assert profile.synopsis_first_sentence == (
        "A catgirl comedy with no closing punctuation"
    )


@pytest.mark.parametrize(
    "desc, first",
    [
        ("ヤニは重度の喫煙癖を持つ猫耳の少女。部屋は灰皿とゴミだらけ。",
         "ヤニは重度の喫煙癖を持つ猫耳の少女。"),
        ("タバコがやめられない！それでも今日も吸ってしまう。",
         "タバコがやめられない！"),
        ("彼女は一体何者なのか？誰も知らない。",
         "彼女は一体何者なのか？"),
    ],
    ids=["kuten", "fullwidth-exclaim", "fullwidth-question"],
)
def test_build_profile_splits_japanese_description_at_fullwidth_terminator(
    desc, first, tmp_path,
):
    """A Japanese-language AniList description (fullwidth 。！？
    terminators, no ASCII punctuation) splits at the FIRST fullwidth
    terminator. Without them in _SENTENCE_END_RE the whole description
    becomes the 'first sentence' and context_hint hard-cuts it mid-word
    at 120 chars -- the word-boundary trim is a no-op on spaceless text.

    Seeded defect: drop 。！？ from _SENTENCE_END_RE (back to r"[.!?]")
    -> the second JA sentence rides along -> red.
    """
    client = _FakeClient(_intel(description=desc))
    profile = build_profile(client, 207141, tmp_path)
    assert profile.synopsis_first_sentence == first


def test_build_profile_empty_description_yields_empty_synopsis(tmp_path):
    """No description -> synopsis_first_sentence == '' (context_hint then
    renders tags only).

    Seeded defect: default the missing description to a placeholder
    string -> red.
    """
    client = _FakeClient(_intel(description=""))
    assert build_profile(client, 1, tmp_path).synopsis_first_sentence == ""


# ------------------------------------------------------- ingest: relations

def test_build_profile_keeps_only_anime_relations(tmp_path):
    """Only ANIME relation nodes become Relation entries: the binding
    Relation shape carries no medium field, and its only consumer (the
    relations carry) is ANIME-only, so MANGA/other nodes are dropped at
    ingest. Live Yani Neko shape: an ADAPTATION MANGA plus a SIDE_STORY
    ANIME; a SIDE_STORY MANGA is the adversarial row (relationType alone
    cannot exclude it).

    Seeded defect: filter on relationType instead of node type -> the
    SIDE_STORY MANGA row appears -> red.
    """
    client = _FakeClient(_intel(relations=[
        {"relationType": "ADAPTATION", "id": 162552, "type": "MANGA",
         "format": "MANGA", "title": "Yani Neko"},
        {"relationType": "SIDE_STORY", "id": 208105, "type": "ANIME",
         "format": "ONA", "title": "Yani Neko Mini"},
        {"relationType": "SIDE_STORY", "id": 999, "type": "MANGA",
         "format": "MANGA", "title": "Yani Neko Anthology"},
    ]))
    profile = build_profile(client, 207141, tmp_path)
    assert profile.relations == [
        Relation("SIDE_STORY", 208105, "Yani Neko Mini")
    ]


# ----------------------------------------------------------- context_hint

def _gits_profile(synopsis: str = "The year is 2029.") -> SeriesProfile:
    # GITS 2026 (177699) live tag spread, deliberately NOT rank-ordered so
    # the composer's own sort is load-bearing. Female Protagonist (84,
    # Cast-Main Cast) outranks three included tags but its category fails
    # the prefix filter; Politics (65) qualifies but is 6th; Gore (20) is
    # under threshold; Seinen's Demographic category never qualifies.
    return SeriesProfile(
        media_id=177699,
        genres=["Action", "Sci-Fi"],
        tags=[
            Tag("Guns", 79, "Theme-Action"),
            Tag("Cyberpunk", 92, "Theme-Sci-Fi"),
            Tag("Female Protagonist", 84, "Cast-Main Cast"),
            Tag("Police", 66, "Theme-Other-Organisations"),
            Tag("Cyborg", 87, "Cast-Traits"),
            Tag("Politics", 65, "Theme-Other"),
            Tag("Seinen", 64, "Demographic"),
            Tag("Artificial Intelligence", 83, "Cast-Traits"),
            Tag("Gore", 20, "Theme-Other"),
        ],
        synopsis_first_sentence=synopsis,
        relations=[],
    )


def test_context_hint_composes_gits_tags_rank_descending_then_synopsis():
    """GITS-2026-shaped profile composes '<tags>; <synopsis>': top-5
    qualifying tags (rank >= 60, Setting-/Theme-/Cast-Traits categories)
    rank-descending, then the synopsis first sentence.

    Seeded defect (the mandated drill): flip the rank threshold to
    <= 60 -> only Gore (20) qualifies -> red. Also red on: dropping the
    rank sort (fixture is shuffled), dropping the category prefix filter
    (Female Protagonist 84 would enter), or lifting the max-5 cap
    (Politics 65 would enter).
    """
    assert context_hint(_gits_profile()) == (
        "Cyberpunk, Cyborg, Artificial Intelligence, Guns, Police; "
        "The year is 2029."
    )


def test_context_hint_never_emits_sexual_content_tags_even_at_high_rank():
    """Yani-Neko-shaped profile: Sexual Content category tags (Scat 62
    live; Watersports raised above threshold as the adversarial case)
    never appear, whatever their rank.

    Seeded defect: filter on rank alone (drop the category rules) ->
    'Watersports' (85) and 'Scat' (62) enter the hint -> red.
    """
    profile = SeriesProfile(
        media_id=207141,
        genres=["Comedy", "Slice of Life"],
        tags=[
            Tag("Drugs", 97, "Theme-Other"),
            Tag("Watersports", 85, "Sexual Content"),
            Tag("Nekomimi", 90, "Cast-Traits"),
            Tag("Scat", 62, "Sexual Content"),
            Tag("Surreal Comedy", 96, "Theme-Comedy"),
            Tag("Urban", 81, "Setting-Scene"),
            Tag("Kemonomimi", 89, "Cast-Traits"),
        ],
        synopsis_first_sentence=_YANI_FIRST_SENTENCE,
        relations=[],
    )
    hint = context_hint(profile)
    assert "Scat" not in hint and "Watersports" not in hint
    assert hint == (
        "Drugs, Surreal Comedy, Nekomimi, Kemonomimi, Urban; "
        + _YANI_FIRST_SENTENCE
    )


def test_context_hint_rank_threshold_includes_60_excludes_59():
    """The threshold is inclusive: rank 60 qualifies, 59 does not.

    Seeded defect: '>= 60' -> '> 60' -> the rank-60 half red; drop the
    threshold entirely -> the rank-59 half red.
    """
    at_60 = SeriesProfile(1, [], [Tag("Urban", 60, "Setting-Scene")], "", [])
    at_59 = SeriesProfile(1, [], [Tag("Urban", 59, "Setting-Scene")], "", [])
    assert context_hint(at_60) == "Urban"
    assert context_hint(at_59) == ""


def test_context_hint_hard_caps_at_120_chars_on_a_word_boundary():
    """A hint over 120 chars truncates at 120, then backs up to the last
    word boundary (no mid-word cut, no ellipsis).

    Seeded defect: drop the cap -> len 146 -> red. Or hard-cut at 120
    without the word-boundary backup -> the dangling 'r' of 'ruins'
    survives -> red.
    """
    profile = SeriesProfile(
        media_id=1,
        genres=[],
        tags=[
            Tag("Post-Apocalyptic Wasteland Survival", 95, "Setting-Universe"),
            Tag("Military Science Fiction", 90, "Theme-Sci-Fi"),
            Tag("Transforming Robots", 85, "Cast-Traits"),
        ],
        synopsis_first_sentence=(
            "Humanity clings to survival in the ruins of a shattered world."
        ),
        relations=[],
    )
    hint = context_hint(profile)
    assert len(hint) <= 120
    assert hint == (
        "Post-Apocalyptic Wasteland Survival, Military Science Fiction, "
        "Transforming Robots; Humanity clings to survival in the"
    )


def test_context_hint_empty_profile_returns_empty_string():
    """No qualifying tags AND no synopsis -> '' (the pipeline then sends
    no Series: line at all).

    Seeded defect: return the '; ' separator unconditionally -> red.
    """
    empty = SeriesProfile(1, [], [], "", [])
    assert context_hint(empty) == ""


def test_context_hint_omits_separator_when_either_part_is_missing():
    """Tags-only and synopsis-only profiles render their one part without
    a dangling '; '.

    Seeded defect: always join 'tags; synopsis' (empty side included) ->
    a leading or trailing separator appears -> red.
    """
    tags_only = SeriesProfile(1, [], [Tag("Nekomimi", 90, "Cast-Traits")], "", [])
    syn_only = SeriesProfile(1, [], [], _YANI_FIRST_SENTENCE, [])
    assert context_hint(tags_only) == "Nekomimi"
    assert context_hint(syn_only) == _YANI_FIRST_SENTENCE


# ------------------------------------------------------------- disk cache

def test_build_profile_round_trips_through_load_profile(tmp_path):
    """build_profile persists profile-<media_id>.json atomically (no .tmp
    litter) and load_profile reads back an EQUAL SeriesProfile.

    Seeded defect: serialize tags as bare names (drop rank/category) ->
    the reloaded profile's tags differ -> equality red. Or point
    load_profile at '<media_id>.json' -> None -> red.
    """
    client = _FakeClient(_intel(
        description=_YANI_DESCRIPTION,
        genres=["Comedy", "Slice of Life"],
        tags=[_tag_row("Nekomimi", 90, "Cast-Traits")],
        relations=[{"relationType": "SIDE_STORY", "id": 208105,
                    "type": "ANIME", "format": "ONA",
                    "title": "Yani Neko Mini"}],
    ))
    built = build_profile(client, 207141, tmp_path)
    assert client.intel_calls == [207141]
    assert (tmp_path / "profile-207141.json").is_file()
    assert list(tmp_path.glob("*.tmp")) == [], (
        "atomic_write_text should have os.replace'd the tmp file away"
    )
    loaded = load_profile(207141, tmp_path)
    assert loaded == built
    assert loaded.tags == [Tag("Nekomimi", 90, "Cast-Traits")]
    assert loaded.relations == [Relation("SIDE_STORY", 208105, "Yani Neko Mini")]
    assert loaded.synopsis_first_sentence == _YANI_FIRST_SENTENCE


def test_load_profile_missing_file_returns_none(tmp_path):
    """No cached profile -> None (no network fallback, no raise).

    Seeded defect: let read_text's FileNotFoundError escape (drop the
    is_file gate and the except) -> red.
    """
    assert load_profile(424242, tmp_path) is None


@pytest.mark.parametrize(
    "raw, why",
    [
        ("{not valid json", "corrupt JSON"),
        ("[1, 2, 3]", "non-dict JSON"),
        ('{"media_id": 207141}', "missing fields"),
        ('{"media_id": 207141, "genres": [], "tags": [{"name": "X"}], '
         '"synopsis_first_sentence": "", "relations": []}',
         "tag row missing rank/category"),
    ],
)
def test_load_profile_corrupt_file_warns_quarantines_and_returns_none(
    raw, why, tmp_path, caplog,
):
    """A corrupt/mistyped cache file logs a WARNING, returns None ('no
    hint for this series', never a crash), and is QUARANTINED aside to
    .json.corrupt.bak, mirroring SeriesGazetteer.load. The quarantine is
    load-bearing: the tray's pick thread only writes a profile it finds
    MISSING (cache-miss build / backfill are both existence-gated), so a
    corrupt file left in place would satisfy the is_file() gate on every
    re-pick and the series hint would stay dead forever.

    Seeded defect: drop the try/except in load_profile -> this raises
    instead of returning None -> red. Drop the _quarantine call (return
    None leaving the file in place) -> the file-gone assert red. Drop
    the warning -> the caplog assert red.
    """
    path = tmp_path / "profile-207141.json"
    path.write_text(raw, encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert load_profile(207141, tmp_path) is None, f"expected None for {why}"
    assert any("profile" in r.message.lower() for r in caplog.records), (
        "corrupt profile cache was not warned about"
    )
    assert not path.exists(), (
        f"corrupt profile ({why}) left in place: every re-pick keeps "
        "taking the cache-hit path and the profile can never heal"
    )
    backup = tmp_path / "profile-207141.json.corrupt.bak"
    assert backup.is_file() and backup.read_text(encoding="utf-8") == raw, (
        "quarantine must move the corrupt bytes aside (not delete them)"
    )


def test_profile_path_defaults_beside_the_gazetteer_cache(tmp_path, monkeypatch):
    """With no cache_dir, the profile lives in the SAME directory as the
    series gazetteer (<app_data_dir>/gazetteers/profile-<id>.json) --
    one cache-root convention, resolved via paths.app_data_dir(), never
    cwd (hard rule #8). build_profile and load_profile agree on it.

    Seeded defect: default to a second root (e.g. app_data_dir()/
    'profiles' or Path('.')) -> the YAMABIKO_APPDATA override + the
    gazetteers/ equality assert red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    assert profile_path(207141) == tmp_path / "gazetteers" / "profile-207141.json"
    client = _FakeClient(_intel(description=_YANI_DESCRIPTION))
    built = build_profile(client, 207141)
    assert (tmp_path / "gazetteers" / "profile-207141.json").is_file()
    assert load_profile(207141) == built


def test_profile_cache_file_is_utf8_json_with_readable_layout(tmp_path):
    """The cache is UTF-8 JSON whose top-level keys mirror the dataclass
    fields -- the on-disk contract load_profile parses.

    Seeded defect: dump with a different shape (e.g. {'profile': ...})
    -> the key asserts red.
    """
    client = _FakeClient(_intel(description=_YANI_DESCRIPTION,
                                genres=["Comedy"]))
    build_profile(client, 207141, tmp_path)
    data = json.loads(
        (tmp_path / "profile-207141.json").read_text(encoding="utf-8"))
    assert set(data) == {"media_id", "genres", "tags",
                         "synopsis_first_sentence", "relations"}
    assert data["media_id"] == 207141
    assert data["genres"] == ["Comedy"]
