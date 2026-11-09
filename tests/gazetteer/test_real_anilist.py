"""Real-tier: the live AniList GraphQL endpoint + a live poster fetch.

Hard rule #3 -- the AniListClient and the palette cover-fetch are external
call sites; unit tests mock httpx, so these exercise the real endpoint.

Skip-vs-fail boundary (M-15 + the 2026-07-11 review fix): skips are for
AVAILABILITY only -- transport failures (offline machine, connection reset),
the client's own outage taxonomy (AniListUnavailable), 5xx server errors, and
a 429 rate limit. Everything else FAILS: a GraphQL 4xx from raise_for_status
is a product defect in OUR query string (charter Sec 8: no conditional skips
based on the thing being tested), and a SUCCESSFUL (2xx) response that parses
to empty hits / characters / cover-URL for _QUERY -- a title that certainly
exists on AniList -- means a GraphQL schema / field-name drift silently
emptied the result (e.g. the Page.media or Media.characters path moved),
exactly the regression this real tier exists to catch.
"""
from __future__ import annotations

import httpx
import pytest

from yamabiko.gazetteer.anilist import AniListClient, AniListUnavailable
from yamabiko.gazetteer.gazetteer import SeriesGazetteer
from yamabiko.overlay_app.series_palette import _fetch_cover_bytes

pytestmark = pytest.mark.real

# A stable, extremely popular franchise whose protagonist リムル is the top
# FAVOURITES_DESC character on every season's roster (so page 1 carries it),
# and whose AniList native name リムル・テンペスト the gazetteer splits into
# リムル -> Rimuru. Used to prove the live search -> fetch -> gazetteer chain.
_QUERY = "Tensei Shitara Slime Datta Ken"

# A verified-public AniList user (checked 2026-07-10); exercises the
# unauthenticated MediaListCollection path without needing a token.
_PUBLIC_USER = "Mand3012"


def _skip_if_unavailable(exc: Exception, what: str) -> None:
    """Skip for genuine availability failures ONLY; re-raise (FAIL) a 4xx.

    httpx.HTTPStatusError is a subclass of httpx.HTTPError, so catching the
    broad base turned a GraphQL 400 from raise_for_status (a PRODUCT defect
    in the query string we composed) into a perpetual skip that could never
    go red. Transport errors, the client's own outage taxonomy
    (AniListUnavailable), 5xx server errors, and a 429 rate limit mean the
    endpoint is unavailable to us right now; any other 4xx means OUR request
    is wrong.
    """
    if (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code < 500
        and exc.response.status_code != 429
    ):
        raise exc
    pytest.skip(f"AniList {what}: {type(exc).__name__}: {exc}")


def _search_or_skip(cli: AniListClient, query: str, limit: int = 8):
    try:
        return cli.search_series(query, limit=limit)
    except (httpx.TransportError, httpx.HTTPStatusError, AniListUnavailable) as exc:
        _skip_if_unavailable(exc, "unreachable")


def test_live_search_returns_hits_with_cover_urls():
    with httpx.Client(timeout=15.0) as c:
        hits = _search_or_skip(AniListClient(client=c), _QUERY)
    # A SUCCESSFUL search for a title that certainly exists must return hits.
    # _search_or_skip already skipped every transport / rate-limit error, so an
    # empty list here is a parsed-empty 2xx: schema/field drift, not "no data".
    assert hits, (
        f"live AniList search for {_QUERY!r} returned zero hits on a 2xx "
        f"response; the query parsed empty -- likely a GraphQL schema drift "
        f"(the Page.media path moved/renamed), not a genuinely missing title"
    )
    # A real search yields real media ids and at least one poster URL.
    assert all(isinstance(h.media_id, int) and h.media_id > 0 for h in hits)
    assert any(h.cover_image and h.cover_image.startswith("http") for h in hits), (
        "no hit carried a coverImage.medium URL on a 2xx search response; the "
        "coverImage field likely drifted in the AniList schema"
    )


# Sousou no Frieren (2023): finished airing with 28 episodes (checked live
# 2026-11-01), so its count is stable.
_FRIEREN_ID = 154587


def test_real_search_returns_episode_counts_within_six_rows():
    """_SEARCH_Q's ``episodes`` field round-trips against the live endpoint,
    and the palette's six-row request reaches AniList: "Gundam", a title
    with dozens of anime (eight hits with the client's default limit,
    checked live 2026-11-01), comes back with exactly six.

    Seeded defects: misspell 'episodes' in _SEARCH_Q -> GraphQL 400 ->
    raise_for_status red; drop the episodes mapping -> None -> red; send
    the default eight instead of ``limit`` -> eight Gundam hits -> red.
    """
    with httpx.Client(timeout=15.0) as c:
        cli = AniListClient(client=c)
        hits = _search_or_skip(cli, "Sousou no Frieren", limit=6)
        gundam = _search_or_skip(cli, "Gundam", limit=6)
    assert len(gundam) == 6, f"a six-row search for Gundam returned {len(gundam)} hits"
    frieren = next((h for h in hits if h.media_id == _FRIEREN_ID), None)
    assert frieren is not None, (
        f"media {_FRIEREN_ID} missing from a 2xx search for its own title: "
        f"{[h.media_id for h in hits]}"
    )
    assert frieren.episodes == 28


def test_live_search_then_characters_builds_gazetteer_with_protagonist():
    with httpx.Client(timeout=15.0) as c:
        cli = AniListClient(client=c)
        hits = _search_or_skip(cli, _QUERY)
        assert hits, (
            f"live AniList search for {_QUERY!r} returned zero hits on a 2xx "
            f"response; parsed-empty search = GraphQL schema drift (Page.media "
            f"path moved), not a missing title"
        )
        # Pick the franchise hit (romaji should contain the slug); fall back to
        # the first hit if titling differs.
        target = next(
            (h for h in hits if "Tensei Shitara Slime" in (h.romaji or "")),
            hits[0],
        )
        try:
            # 2 pages is plenty -- リムル is the #1 favourite (page 1). Caps
            # request count + rate-limit exposure.
            shares: list[float] = []
            chars = cli.fetch_characters(
                target.media_id, max_pages=2, on_page=shares.append)
        except (httpx.TransportError, httpx.HTTPStatusError,
                AniListUnavailable) as exc:
            _skip_if_unavailable(exc, "characters unreachable")
    # Every transport / outage / rate-limit error skipped above. An empty
    # list on a 2xx means the Media.characters nodes path drifted -- a
    # schema regression, not "no characters".
    assert chars, (
        f"live AniList fetch_characters(media_id={target.media_id}) returned "
        f"zero characters on a 2xx response; the Media.characters.nodes path "
        f"likely drifted in the GraphQL schema"
    )
    # The live pageInfo drives on_page: this cast is far longer than one page
    # of 25, so page 1 reports half the 2-page cap and page 2 (the cap) the
    # full share. A drifted hasNextPage makes page 1 the last page ([1.0]).
    assert shares == [0.5, 1.0], f"live page shares {shares}"
    gaz = SeriesGazetteer.from_characters(chars)
    # The real end-to-end payoff: the protagonist's katakana maps to a
    # Rimuru-containing official English spelling, built from live data.
    en = gaz.lookup("リムル")
    assert en is not None, f"リムル not in gazetteer of {len(chars)} chars for media_id={target.media_id}"
    assert "Rimuru" in en


def test_real_surname_first_orientation_rezero():
    # Re:Zero (AniList media 21355): Natsuki Subaru's native is surname-first
    # (ナツキ・スバル), full is "Subaru Natsuki". Guards the orientation fix
    # against AniList convention drift.
    try:
        chars = AniListClient().fetch_characters(21355)
    except AniListUnavailable:
        pytest.skip("AniList degraded/outage")
    gaz = SeriesGazetteer.from_characters(chars)
    assert gaz.lookup("スバル") == "Subaru"
    assert gaz.lookup("ナツキ") == "Natsuki"


def test_live_cover_fetch_returns_image_bytes():
    with httpx.Client(timeout=15.0) as c:
        hits = _search_or_skip(AniListClient(client=c), _QUERY)
    url = next((h.cover_image for h in hits if h.cover_image), None) if hits else None
    # A 2xx search for a title that certainly exists must yield at least one
    # cover URL. Its absence (empty hits, or every hit missing coverImage.medium)
    # is schema drift, not a transport failure -- those already skipped above.
    assert url, (
        f"live AniList search for {_QUERY!r} produced no coverImage.medium URL "
        f"on a 2xx response; the coverImage/media path likely drifted in the "
        f"GraphQL schema"
    )
    data = _fetch_cover_bytes(url)
    if data is None:
        # _fetch_cover_bytes returns None on ANY CDN/network failure (non-200,
        # timeout, reset) -- a genuine transport problem, so skip. A 200 with an
        # empty/HTML body returns bytes that fail the magic-number asserts below.
        pytest.skip(f"cover fetch returned None (CDN/network): {url}")
    # A real poster is a non-trivial JPEG/PNG/WebP; assert a plausible size +
    # a known image magic number so a 200-with-HTML-error-page can't pass.
    assert len(data) > 500
    assert data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:4] == b"RIFF"


def test_real_fetch_media_intel_returns_ranked_tags_with_nekomimi():
    """_INTEL_Q round-trips against the live endpoint: media 207141 returns
    a non-empty ranked tag list including its known 'Nekomimi' tag.

    Seeded defect: misspell any _INTEL_Q field (e.g. 'tag' for 'tags') ->
    GraphQL 400 -> raise_for_status red; or drop the tags mapping in
    fetch_media_intel -> the non-empty assert red.
    """
    try:
        intel = AniListClient().fetch_media_intel(207141)
    except AniListUnavailable:
        pytest.skip("AniList degraded/outage")
    assert intel["tags"], "media 207141 returned no tags"
    names = [t.get("name") for t in intel["tags"]]
    assert "Nekomimi" in names, f"'Nekomimi' tag missing from {names}"
    # Tag rows must carry the fields the profile layer filters on.
    assert all("rank" in t and "isMediaSpoiler" in t for t in intel["tags"])


def test_real_fetch_watching_public_user_returns_hits_without_raising():
    """_WATCHING_Q round-trips unauthenticated against a verified-public
    user's CURRENT list and every row maps onto a valid SeriesHit. The
    list MAY legitimately be empty (nothing being watched); the claim is
    the wire path completes and rows are well-formed, not list length.
    GraphQL validates every requested field whatever the data, so this also
    pins the entry's ``updatedAt`` and the media's ``episodes``.

    Seeded defect: misspell any _WATCHING_Q field (e.g. 'entrys' or
    'updatedAtt') -> GraphQL 400 -> raise_for_status red; or map media ids
    as strings -> the per-row media_id assert red.
    """
    try:
        hits = AniListClient().fetch_watching(_PUBLIC_USER)
    except AniListUnavailable:
        pytest.skip("AniList degraded/outage")
    assert isinstance(hits, list)
    for h in hits:
        assert isinstance(h.media_id, int) and h.media_id > 0
        assert isinstance(h.romaji, str)
        assert h.episodes is None or (isinstance(h.episodes, int) and h.episodes > 0)
