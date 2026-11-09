"""Tests for yamabiko.gazetteer.anilist -- the AniList GraphQL wire client.

Unit tier only here, per docs/TESTING.md Sec 4: httpx.MockTransport
fakes the network; verifies response parsing, the 429/403/transport
error taxonomy, pacing, and the bearer-token seam. Sleeps are injected
(``AniListClient(sleep=...)``) so no test ever really sleeps. Fast, no
network, runs by default.

The live-AniList @pytest.mark.real coverage (hard rule #3) lives in
tests/gazetteer/test_real_anilist.py against the real
https://graphql.anilist.co endpoint.
"""
from __future__ import annotations

import json

import httpx
import pytest

from yamabiko.gazetteer.anilist import (
    AniListClient,
    AniListPrivateUser,
    AniListUnavailable,
)


def _client(handler, **kw):
    return AniListClient(httpx.Client(transport=httpx.MockTransport(handler)), **kw)


def _search_payload():
    return {"data": {"Page": {"media": [
        {"id": 9756, "title": {"romaji": "Mahou Shoujo Madoka Magica",
         "english": "Puella Magi Madoka Magica", "native": "魔法少女まどか☆マギカ"},
         "seasonYear": 2011, "format": "TV",
         "coverImage": {"medium": "https://img.anilist.co/madoka.jpg",
                        "color": "#e4a15d"}}]}}}


# ---------------------------------------------------------------- parsing

def test_fetch_characters_walks_pages():
    """Follows pageInfo.hasNextPage across pages and flattens edge nodes.

    Seeded defect: ignore hasNextPage (fetch page 1 only) -> the page-2
    character is missing -> the 'Benimaru' assert red.
    """
    pages = {1: (True, [{"role": "MAIN",
                         "node": {"name": {"native": "リムル・テンペスト",
                                           "full": "Rimuru Tempest",
                                           "alternative": ["Satoru Mikami"]}}}]),
             2: (False, [{"role": "SUPPORTING",
                          "node": {"name": {"native": "ベニマル", "full": "Benimaru",
                                            "alternative": []}}}])}
    def handler(req: httpx.Request) -> httpx.Response:
        page = req.read().decode(); pnum = 2 if '"page":2' in page or '"page": 2' in page else 1
        nxt, edges = pages[pnum]
        return httpx.Response(200, json={"data": {"Media": {"characters": {
            "pageInfo": {"hasNextPage": nxt}, "edges": edges}}}})
    chars = _client(handler, sleep=lambda s: None).fetch_characters(9756)
    natives = [c.native for c in chars]
    assert "リムル・テンペスト" in natives and "ベニマル" in natives
    assert chars[0].full == "Rimuru Tempest"


def test_fetch_characters_parses_role_from_edges():
    """characters.edges.role (MAIN / BACKGROUND) rides onto Character.role,
    which the gazetteer build later uses to drop generic-noun BACKGROUND
    entries.

    Seeded defect: stop reading edge['role'] (leave the dataclass default
    None) -> both role asserts red.
    """
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"Media": {"characters": {
            "pageInfo": {"hasNextPage": False},
            "edges": [
                {"role": "MAIN", "node": {"name": {"native": "素子",
                 "full": "Motoko Kusanagi", "alternative": []}}},
                {"role": "BACKGROUND", "node": {"name": {"native": "店員",
                 "full": None, "alternative": []}}},
            ]}}}})
    chars = _client(handler).fetch_characters(1)
    assert [c.role for c in chars] == ["MAIN", "BACKGROUND"]
    assert chars[0].full == "Motoko Kusanagi"


def test_search_series_parses_media_hits():
    """The GraphQL Page.media list maps to SeriesHit rows (id + titles +
    cover art).

    Seeded defect: read response['data']['Media'] (singular) instead of
    Page.media -> KeyError/empty -> the len/assert goes red. Or drop the
    coverImage parse (leave cover_image/color at their None default) ->
    the two cover-art asserts go red.
    """
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_search_payload())
    hits = _client(handler).search_series("madoka")
    assert len(hits) == 1
    assert hits[0].media_id == 9756
    assert hits[0].english == "Puella Magi Madoka Magica"
    assert hits[0].romaji.startswith("Mahou")
    assert hits[0].cover_image == "https://img.anilist.co/madoka.jpg"
    assert hits[0].color == "#e4a15d"


def test_search_series_maps_genres_adult_and_airing_fields():
    """genres / isAdult / nextAiringEpisode map onto the appended SeriesHit
    fields, and the outgoing query actually requests them (synonyms
    included -- it rides the same request for later consumers).

    Seeded defect: drop 'genres synonyms isAdult' or the nextAiringEpisode
    block from _SEARCH_Q -> the request-body asserts red; or drop the
    isAdult/nextAiringEpisode mapping in search_series -> the field
    asserts red.
    """
    seen: dict = {}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["query"] = json.loads(req.read().decode())["query"]
        payload = _search_payload()
        payload["data"]["Page"]["media"][0].update(
            {"genres": ["Drama", "Psychological"], "isAdult": True,
             "synonyms": ["Madoka"],
             "nextAiringEpisode": {"episode": 11, "airingAt": 1760000000}})
        return httpx.Response(200, json=payload)
    hits = _client(handler).search_series("madoka")
    for field in ("genres", "synonyms", "isAdult", "nextAiringEpisode"):
        assert field in seen["query"], f"_SEARCH_Q no longer requests {field}"
    assert hits[0].genres == ["Drama", "Psychological"]
    assert hits[0].is_adult is True
    assert hits[0].next_airing_episode == 11
    assert hits[0].next_airing_at == 1760000000


def test_search_series_returns_no_hits_when_page_is_null_in_a_200():
    """AniList can answer HTTP 200 with {"data": {"Page": null},
    "errors": [...]} -- the key PRESENT, its value null. search_series
    must degrade to zero hits, never crash the worker.

    Seeded defect: parse Page with ``.get("Page", {})`` (the dict default
    fires only when the KEY is absent, not when its value is null) ->
    ``None.get("media")`` raises AttributeError -> red. (This IS the
    pre-fix code; confirmed red against it.)
    """
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "data": {"Page": None},
            "errors": [{"message": "Something went wrong", "status": 500}]})
    assert _client(handler).search_series("madoka") == []


def test_search_series_tolerates_missing_optional_blocks():
    """A media row with no coverImage / genres / nextAiringEpisode blocks
    parses with those fields at their defaults -- the palette then renders
    the neutral color tile.

    Seeded defect: parse coverImage as ``m["coverImage"]`` or
    nextAiringEpisode as ``m["nextAiringEpisode"]`` (no ``or {}`` guard)
    -> KeyError on this row -> red.
    """
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"Page": {"media": [
            {"id": 1, "title": {"romaji": "No Cover", "english": None,
             "native": None}, "seasonYear": None, "format": "TV"}]}}})
    hits = _client(handler).search_series("nocover")
    assert len(hits) == 1
    assert hits[0].cover_image is None
    assert hits[0].color is None
    assert hits[0].is_adult is False
    assert hits[0].genres is None
    assert hits[0].next_airing_episode is None
    assert hits[0].next_airing_at is None


# ------------------------------------------------- 429 / outage taxonomy

def test_429_then_200_sleeps_retry_after_then_returns_hits():
    """One 429 with Retry-After triggers exactly one injected sleep of the
    header value, one retry, and the retried response is parsed normally.

    Seeded defect: skip the retry (raise on the first 429) -> this test
    red (AniListUnavailable escapes instead of hits returning).
    """
    calls = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json=_search_payload())
    sleeps: list[float] = []
    hits = _client(handler, sleep=sleeps.append).search_series("madoka")
    assert [h.media_id for h in hits] == [9756]
    assert sleeps == [7]
    assert calls["n"] == 2


def test_429_retry_after_above_cap_sleeps_65():
    """Retry-After beyond the 65s cap is clamped -- an adversarial/stuck
    header must not stall a pick for 10 minutes.

    Seeded defect: drop the min(retry_after, 65) clamp -> sleeps == [600]
    -> red.
    """
    calls = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "600"})
        return httpx.Response(200, json=_search_payload())
    sleeps: list[float] = []
    _client(handler, sleep=sleeps.append).search_series("madoka")
    assert sleeps == [65]


def test_429_negative_retry_after_clamps_to_zero_sleep():
    """A negative/adversarial Retry-After must clamp to a 0s sleep (then
    the single retry proceeds) -- with the real time.sleep, a negative
    value raises ValueError, which would escape as a crash instead of
    the promised AniListUnavailable-or-success taxonomy.

    Seeded defect: drop the max(0, ...) lower clamp -> sleeps == [-5]
    -> red (and the real time.sleep(-5) raises 'ValueError: sleep
    length must be non-negative').
    """
    calls = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "-5"})
        return httpx.Response(200, json=_search_payload())
    sleeps: list[float] = []
    hits = _client(handler, sleep=sleeps.append).search_series("madoka")
    assert sleeps == [0]
    assert [h.media_id for h in hits] == [9756]


def test_missing_retry_after_header_defaults_to_60():
    """A 429 without Retry-After sleeps the documented 60s default.

    Seeded defect: default the header read to 0 -> sleeps == [0] -> red.
    """
    calls = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=_search_payload())
    sleeps: list[float] = []
    _client(handler, sleep=sleeps.append).search_series("madoka")
    assert sleeps == [60]


def test_second_consecutive_429_raises_unavailable_after_one_retry():
    """Two consecutive 429s raise AniListUnavailable after exactly two
    POSTs -- retry ONCE, never loop.

    Seeded defect: keep retrying on every 429 -> the request count passes
    2 (and nothing raises) -> both asserts red.
    """
    calls = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "1"})
    sleeps: list[float] = []
    with pytest.raises(AniListUnavailable):
        _client(handler, sleep=sleeps.append).search_series("madoka")
    assert calls["n"] == 2
    assert sleeps == [1]


def test_http_403_raises_unavailable():
    """HTTP 403 is AniList's documented whole-API outage mode; it maps to
    AniListUnavailable so UI layers render 'temporarily unavailable',
    never a traceback.

    Seeded defect: remove the 403 branch (fall through to
    raise_for_status) -> httpx.HTTPStatusError escapes -> red.
    """
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(403)
    with pytest.raises(AniListUnavailable):
        _client(handler).search_series("madoka")


def test_transport_error_raises_unavailable():
    """Connect/timeout failures (httpx.TransportError) map to
    AniListUnavailable -- an offline machine is an 'AniList unavailable'
    condition, not a crash.

    Seeded defect: drop the except-TransportError mapping in the POST
    helper -> httpx.ConnectError escapes -> red.
    """
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to graphql.anilist.co")
    with pytest.raises(AniListUnavailable):
        _client(handler).search_series("madoka")


# ----------------------------------------------------------------- pacing

def test_fetch_characters_sleeps_quarter_second_between_pages_only():
    """Three character pages produce exactly two injected 0.25s sleeps --
    between requests, never before the first (burst-limiter headroom).

    Seeded defect: remove the inter-page sleep -> sleeps == [] -> red.
    (Sleeping before page 1 instead -> 3 entries -> also red.)
    """
    def handler(req: httpx.Request) -> httpx.Response:
        page = json.loads(req.read().decode())["variables"]["page"]
        return httpx.Response(200, json={"data": {"Media": {"characters": {
            "pageInfo": {"hasNextPage": page < 3},
            "edges": [{"role": "MAIN", "node": {"name": {
                "native": f"キャラ{page}", "full": None, "alternative": []}}}]}}}})
    sleeps: list[float] = []
    chars = _client(handler, sleep=sleeps.append).fetch_characters(1, max_pages=6)
    assert len(chars) == 3
    assert sleeps == [0.25, 0.25]


def test_fetch_characters_reports_each_page_and_the_full_share_on_the_last():
    """on_page hears the share of the cast fetched after every page:
    page / max_pages, and 1.0 on the last page (no next page), so the pick's
    progress line reaches the end of the cast stage even when the cast is
    shorter than max_pages.

    Seeded defects: report page / max_pages on the last page too -> the last
    share is 0.5 -> red; never call on_page -> [] -> red.
    """
    def handler(req: httpx.Request) -> httpx.Response:
        page = json.loads(req.read().decode())["variables"]["page"]
        return httpx.Response(200, json={"data": {"Media": {"characters": {
            "pageInfo": {"hasNextPage": page < 3}, "edges": []}}}})

    shares: list[float] = []
    _client(handler, sleep=lambda s: None).fetch_characters(
        1, max_pages=6, on_page=shares.append)
    assert shares == pytest.approx([1 / 6, 2 / 6, 1.0])


# ------------------------------------------------------------- token seam

def test_token_sends_bearer_authorization_on_post():
    """When a token is configured, the POST carries
    Authorization: Bearer <token> (the OAuth-ready seam).

    Seeded defect: stop passing headers in the POST helper -> the captured
    header is None -> red.
    """
    seen: dict = {}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("Authorization")
        return httpx.Response(200, json=_search_payload())
    _client(handler, token="t0k3n").search_series("madoka")
    assert seen["auth"] == "Bearer t0k3n"


def test_no_token_sends_no_authorization_header():
    """token=None must not invent an Authorization header (AniList rejects
    malformed bearers, and unauthenticated is the shipped posture).

    Seeded defect: always attach 'Bearer None' -> header present -> red.
    """
    seen: dict = {}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["has_auth"] = "authorization" in req.headers
        return httpx.Response(200, json=_search_payload())
    _client(handler).search_series("madoka")
    assert seen["has_auth"] is False


# ------------------------------------------------------------------ close


def test_close_closes_the_underlying_httpx_client():
    """AniListClient.close() closes the wrapped httpx.Client: the tray's
    per-pick build/backfill workers call it in a ``finally`` so each
    pick's client releases its connections instead of leaking until GC.

    Seeded defect: make close() a no-op (or delete the method) ->
    ``inner.is_closed`` stays False -> red.
    """
    inner = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={})))
    AniListClient(inner).close()
    assert inner.is_closed


# ------------------------------------------------------- fetch_media_intel

def test_fetch_media_intel_returns_genres_tags_description_relations():
    """_INTEL_Q's response shapes into the raw-ish intel dict: tags pass
    through with spoiler flags intact, relations flatten edge+node with
    the romaji title string.

    Seeded defect: read data.Page instead of data.Media -> every assert
    red. Or drop the relations edge+node flatten -> relations == [] -> red.
    """
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"Media": {
            "genres": ["Action", "Sci-Fi"],
            "tags": [{"name": "Cyberpunk", "rank": 92,
                      "category": "Setting-Universe",
                      "isMediaSpoiler": False, "isGeneralSpoiler": False}],
            "description": "In the future, cyberbrains are hacked.",
            "relations": {"edges": [{"relationType": "PREQUEL",
                "node": {"id": 42, "type": "ANIME", "format": "TV",
                         "title": {"romaji": "Koukaku Kidoutai SAC"}}}]}}}})
    intel = _client(handler).fetch_media_intel(207141)
    assert intel["genres"] == ["Action", "Sci-Fi"]
    assert intel["tags"] == [{"name": "Cyberpunk", "rank": 92,
                              "category": "Setting-Universe",
                              "isMediaSpoiler": False,
                              "isGeneralSpoiler": False}]
    assert intel["description"] == "In the future, cyberbrains are hacked."
    assert intel["relations"] == [{"relationType": "PREQUEL", "id": 42,
                                   "type": "ANIME", "format": "TV",
                                   "title": "Koukaku Kidoutai SAC"}]


# ---------------------------------------------------------- fetch_watching

def _watching_media(media_id: int, romaji: str, **extra) -> dict:
    m = {"id": media_id, "title": {"romaji": romaji, "english": None,
                                   "native": None},
         "seasonYear": 2026, "format": "TV",
         "coverImage": {"medium": f"https://img.anilist.co/{media_id}.jpg",
                        "color": "#111111"}}
    m.update(extra)
    return m


def test_fetch_watching_flattens_custom_lists_and_dedupes_by_media_id():
    """Entries across ALL returned lists (custom lists included) flatten
    into SeriesHits deduped by media id; airing fields ride along. Media
    333 lives ONLY in the second (custom) list, so the flatten half of
    the claim is genuinely load-bearing, not just the dedupe half.

    Seeded defect: read only lists[0] (``for lst in lists[:1]``) ->
    media 333 missing -> red. Or drop the seen-set dedupe -> 111
    appears twice -> the media-id list assert red.
    """
    def handler(req: httpx.Request) -> httpx.Response:
        dup = _watching_media(111, "Watching A",
                              nextAiringEpisode={"episode": 5,
                                                 "airingAt": 1760000000})
        return httpx.Response(200, json={"data": {"MediaListCollection": {
            "lists": [
                {"entries": [{"media": dup},
                             {"media": _watching_media(222, "Watching B")}]},
                # Custom list: repeats 111 (dedupe) AND is the only home
                # of 333 (flatten-across-lists).
                {"entries": [{"media": dup},
                             {"media": _watching_media(333, "Watching C")}]},
            ]}}})
    hits = _client(handler).fetch_watching("Mand3012")
    assert [h.media_id for h in hits] == [111, 222, 333]
    assert hits[0].romaji == "Watching A"
    assert hits[0].next_airing_episode == 5
    assert hits[0].next_airing_at == 1760000000
    assert hits[1].next_airing_episode is None
    assert hits[2].romaji == "Watching C"


def test_episodes_ride_both_queries_onto_series_hit():
    """Media.episodes is requested by the search AND the watch-list query and
    lands on SeriesHit.episodes: the series bar's "28 episodes".

    Seeded defects: drop ``episodes`` from either query -> that query's
    text assert red; drop the mapping in _hit_from_media -> both value
    asserts red.
    """
    queries: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        query = json.loads(req.read().decode())["query"]
        queries.append(query)
        if "MediaListCollection" in query:
            return httpx.Response(200, json={"data": {"MediaListCollection": {
                "lists": [{"entries": [
                    {"updatedAt": 1, "media": _watching_media(111, "A", episodes=12)},
                ]}]}}})
        payload = _search_payload()
        payload["data"]["Page"]["media"][0]["episodes"] = 28
        return httpx.Response(200, json=payload)

    client = _client(handler)
    assert client.search_series("madoka")[0].episodes == 28
    assert client.fetch_watching("Mand3012")[0].episodes == 12
    assert len(queries) == 2
    assert all("episodes" in q for q in queries), "a query no longer requests episodes"


def test_fetch_watching_orders_hits_by_last_update_across_lists():
    """Currently watching shows the most recently updated entries first,
    across ALL lists: AniList sorts each list on its own (probe 2026-11-01),
    so the order comes from each entry's updatedAt after the flatten. 333 is
    the newest and lives only in the custom list; 222 repeats there.

    Seeded defects: drop the sort -> list order [111, 222, 333] -> red; sort
    ascending -> red; stop requesting updatedAt -> the query assert red.
    """
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["query"] = json.loads(req.read().decode())["query"]
        return httpx.Response(200, json={"data": {"MediaListCollection": {
            "lists": [
                {"entries": [
                    {"updatedAt": 100, "media": _watching_media(111, "Oldest")},
                    {"updatedAt": 300, "media": _watching_media(222, "Middle")},
                ]},
                {"entries": [
                    {"updatedAt": 500, "media": _watching_media(333, "Newest")},
                    {"updatedAt": 300, "media": _watching_media(222, "Middle")},
                ]},
            ]}}})

    hits = _client(handler).fetch_watching("Mand3012")
    assert "updatedAt" in seen["query"]
    assert [h.media_id for h in hits] == [333, 222, 111]


def test_fetch_watching_private_user_raises_private_user():
    """The live-verified private-list response (HTTP 404 + GraphQL
    'Private User' error) raises AniListPrivateUser so the palette can
    render 'list is private'.

    Seeded defect: raise_for_status before scanning the GraphQL errors ->
    httpx.HTTPStatusError escapes instead -> red.
    """
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={
            "errors": [{"message": "Private User", "status": 404}],
            "data": {"MediaListCollection": None}})
    with pytest.raises(AniListPrivateUser):
        _client(handler).fetch_watching("someone_private")


def test_private_user_error_message_never_carries_the_username():
    """The AniList username is an account identifier and exception
    messages end up in logged reprs and tracebacks (diagnostics zips ship
    with no consent gate), so AniListPrivateUser's MESSAGE must stay a
    fixed string; the username rides only on the ``username`` attribute
    for callers that legitimately need it.

    Seeded defect: pass the username to Exception.__init__ (the pre-fix
    code: the message IS the username) -> str/repr contain it -> red.
    """
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={
            "errors": [{"message": "Private User", "status": 404}],
            "data": {"MediaListCollection": None}})
    with pytest.raises(AniListPrivateUser) as excinfo:
        _client(handler).fetch_watching("SecretWatcher99")
    assert "SecretWatcher99" not in str(excinfo.value)
    assert "SecretWatcher99" not in repr(excinfo.value)
    assert excinfo.value.username == "SecretWatcher99"
