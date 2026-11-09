"""AniList GraphQL client. https://graphql.anilist.co

Reuses httpx (already a runtime dep -- see yamabiko/translation/translator.py).
Rate limit currently 30 req/min (degraded); 90/min nominal; see
docs.anilist.co/guide/rate-limiting. Callers cache + debounce; the client
itself honours one Retry-After per request and paces multi-page fetches
for burst-limiter headroom.

Auth posture is read-only + username-based (no OAuth shipped). The optional
``token`` constructor arg is the OAuth-ready seam: when set, every POST
carries ``Authorization: Bearer <token>``. Nothing sets it today.

Error taxonomy (module-level so UI layers can catch and render a friendly
line, never a traceback):

- ``AniListUnavailable`` -- HTTP 403 (AniList's documented whole-API outage
  mode), a second consecutive 429, or a connect/timeout failure.
- ``AniListPrivateUser`` -- the "Private User" GraphQL error from
  ``fetch_watching``.

Everything else keeps ``raise_for_status`` behavior.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import httpx

_ENDPOINT = "https://graphql.anilist.co"

# Watch-list rows must map onto the same SeriesHit shape as search rows, so
# _WATCHING_Q's media block mirrors this one (parsed by _hit_from_media).
_SEARCH_Q = """
query ($q: String, $n: Int) {
  Page(perPage: $n) {
    media(search: $q, type: ANIME, sort: SEARCH_MATCH) {
      id title { romaji english native } seasonYear format episodes
      coverImage { medium color }
      genres synonyms isAdult
      nextAiringEpisode { episode airingAt }
    }
  }
}
"""

_CHARS_Q = """
query ($id: Int, $page: Int) {
  Media(id: $id) {
    characters(sort: FAVOURITES_DESC, page: $page, perPage: 25) {
      pageInfo { hasNextPage }
      edges { role node { name { native full alternative } } }
    }
  }
}
"""

_INTEL_Q = """
query ($id: Int) {
  Media(id: $id) {
    genres
    tags { name rank category isMediaSpoiler isGeneralSpoiler }
    description(asHtml: false)
    relations { edges { relationType node { id type format title { romaji } } } }
  }
}
"""

_WATCHING_Q = """
query ($u: String) {
  MediaListCollection(userName: $u, type: ANIME, status: CURRENT) {
    lists {
      entries {
        updatedAt
        media {
          id title { romaji english native } seasonYear format episodes
          coverImage { medium color }
          genres synonyms isAdult
          nextAiringEpisode { episode airingAt }
        }
      }
    }
  }
}
"""


class AniListUnavailable(Exception):
    """AniList cannot serve requests right now (outage 403, repeated 429,
    or connect/timeout). Render 'AniList temporarily unavailable'."""


class AniListPrivateUser(Exception):
    """The AniList user's list is private ('Private User' GraphQL error).

    The username is an account identifier and exception messages end up
    in logged reprs and tracebacks, so the MESSAGE is a fixed string;
    the username rides only on the ``username`` attribute for callers
    that legitimately need it.
    """

    def __init__(self, username: str) -> None:
        super().__init__("AniList list is private")
        self.username = username


@dataclass(frozen=True)
class SeriesHit:
    media_id: int
    romaji: str
    english: str | None
    native: str | None
    season_year: int | None
    fmt: str | None
    # Added at the END with defaults so existing positional construction
    # (and every caller that unpacks the first six fields) keeps working.
    # cover_image is the AniList ``coverImage.medium`` URL; color is the
    # cover's dominant colour hex (``coverImage.color``), used as the
    # palette's tile fallback when the poster can't be fetched.
    cover_image: str | None = None
    color: str | None = None
    # Second appended-with-defaults block (same compat rule). genres stays
    # None (vs []) when the query/row didn't carry a genres field, so
    # consumers can tell "no genres" from "not fetched".
    is_adult: bool = False
    genres: list[str] | None = None
    next_airing_episode: int | None = None
    next_airing_at: int | None = None
    # Third appended-with-defaults block (same compat rule): Media.episodes,
    # the planned episode count; None while unknown (a show that has not
    # announced its length). The series bar's "N episodes".
    episodes: int | None = None


@dataclass(frozen=True)
class Character:
    native: str | None
    full: str | None
    alternative: list[str]
    # AniList characters.edges.role: MAIN / SUPPORTING / BACKGROUND.
    # Appended with a default so 3-arg positional construction keeps
    # working. BACKGROUND entries include generic-noun "names" (店員 /
    # 職員), which the gazetteer build filters on this field.
    role: str | None = None


def _hit_from_media(m: dict) -> SeriesHit:
    t = m.get("title") or {}
    ci = m.get("coverImage") or {}
    nae = m.get("nextAiringEpisode") or {}
    return SeriesHit(m["id"], t.get("romaji") or "", t.get("english"),
                     t.get("native"), m.get("seasonYear"), m.get("format"),
                     cover_image=ci.get("medium"), color=ci.get("color"),
                     is_adult=bool(m.get("isAdult")),
                     genres=m.get("genres"),
                     next_airing_episode=nae.get("episode"),
                     next_airing_at=nae.get("airingAt"),
                     episodes=m.get("episodes"))


class AniListClient:
    def __init__(self, client: httpx.Client | None = None,
                 token: str | None = None, sleep=time.sleep) -> None:
        self._client = client or httpx.Client(timeout=15.0)
        self._token = token
        self._sleep = sleep

    def _post(self, query: str, variables: dict) -> httpx.Response:
        """POST with the shared failure taxonomy (applies to every query).

        Transport errors and HTTP 403 raise ``AniListUnavailable``. A 429
        sleeps ``max(0, min(Retry-After, 65))`` (header default 60; the cap
        keeps a stuck/adversarial header from stalling a pick, the floor
        keeps a negative one from crashing ``time.sleep``) and retries ONCE; a
        second consecutive 429 raises ``AniListUnavailable``. Success
        statuses are NOT checked here: callers ``raise_for_status`` after
        any GraphQL-error inspection they need (``fetch_watching`` must
        read the "Private User" error out of a 404 body first).
        """
        for attempt in (0, 1):
            try:
                r = self._client.post(
                    _ENDPOINT,
                    json={"query": query, "variables": variables},
                    headers=({"Authorization": f"Bearer {self._token}"}
                             if self._token else None))
            except httpx.TransportError as exc:
                raise AniListUnavailable(
                    f"AniList unreachable: {type(exc).__name__}: {exc}") from exc
            if r.status_code == 429:
                if attempt:
                    raise AniListUnavailable(
                        "AniList rate limit: two consecutive 429s")
                try:
                    retry_after = int(r.headers.get("Retry-After", "60"))
                except ValueError:
                    retry_after = 60
                self._sleep(max(0, min(retry_after, 65)))
                continue
            if r.status_code == 403:
                raise AniListUnavailable("AniList outage mode (HTTP 403)")
            return r
        raise AniListUnavailable("unreachable")  # pragma: no cover

    def close(self) -> None:
        """Close the underlying httpx.Client (idempotent). The tray's
        per-pick build/backfill workers call this in a ``finally`` so
        each pick's client releases its connections deterministically."""
        self._client.close()

    def search_series(self, query: str, limit: int = 8) -> list[SeriesHit]:
        r = self._post(_SEARCH_Q, {"q": query, "n": limit})
        r.raise_for_status()
        # ``or {}`` (not a .get default): AniList serves 200s with
        # {"data": {"Page": null}, "errors": [...]} -- key present, value
        # null -- and a dict default only fires when the key is ABSENT.
        media = (((r.json().get("data") or {}).get("Page") or {})
                 .get("media")) or []
        return [_hit_from_media(m) for m in media]

    def fetch_characters(self, media_id: int, max_pages: int = 6,
                         on_page: Callable[[float], None] | None = None,
                         ) -> list[Character]:
        """The media's characters, most favourited first, 25 per page, at
        most ``max_pages`` pages. After each page ``on_page`` (the pick's
        progress line) hears the share of the cast fetched: ``page /
        max_pages``, and 1.0 on the last page (no next page, or the cap)."""
        out: list[Character] = []
        for page in range(1, max_pages + 1):
            if page > 1:
                # Between page requests only (never before the first):
                # headroom under AniList's separate burst limiter.
                self._sleep(0.25)
            r = self._post(_CHARS_Q, {"id": media_id, "page": page})
            r.raise_for_status()
            block = (((r.json().get("data") or {}).get("Media") or {})
                     .get("characters") or {})
            for e in block.get("edges", []) or []:
                nm = ((e.get("node") or {}).get("name")) or {}
                out.append(Character(nm.get("native"), nm.get("full"),
                                     list(nm.get("alternative") or []),
                                     role=e.get("role")))
            last = (page == max_pages
                    or not (block.get("pageInfo") or {}).get("hasNextPage"))
            if on_page is not None:
                on_page(1.0 if last else page / max_pages)
            if last:
                break
        return out

    def fetch_media_intel(self, media_id: int) -> dict:
        """One-request series intelligence: genres, ranked tags (spoiler
        flags intact -- the profile layer filters), plain-text description,
        and relations flattened to one dict per edge with the romaji title.
        """
        r = self._post(_INTEL_Q, {"id": media_id})
        r.raise_for_status()
        m = ((r.json().get("data") or {}).get("Media")) or {}
        relations = []
        for e in ((m.get("relations") or {}).get("edges") or []):
            n = e.get("node") or {}
            relations.append({"relationType": e.get("relationType"),
                              "id": n.get("id"), "type": n.get("type"),
                              "format": n.get("format"),
                              "title": (n.get("title") or {}).get("romaji")})
        return {"genres": list(m.get("genres") or []),
                "tags": list(m.get("tags") or []),
                "description": m.get("description") or "",
                "relations": relations}

    def fetch_watching(self, username: str) -> list[SeriesHit]:
        """The user's CURRENT (watching) list, flattened across ALL
        returned lists (custom lists repeat entries), deduped by media id
        (first occurrence wins), newest list update first. AniList orders
        each list on its own (probe 2026-11-01), so the order across lists
        comes from each entry's ``updatedAt`` here; an entry without one goes
        last, and ties keep AniList's order (the sort is stable).
        """
        r = self._post(_WATCHING_Q, {"u": username})
        # A private list is an HTTP 404 whose GraphQL body says "Private
        # User" (live-verified 2026-07-10), so the error scan must run
        # BEFORE raise_for_status.
        try:
            body = r.json()
        except ValueError:
            body = {}
        for err in body.get("errors") or []:
            if "Private User" in str(err.get("message") or ""):
                raise AniListPrivateUser(username)
        r.raise_for_status()
        lists = (((body.get("data") or {}).get("MediaListCollection") or {})
                 .get("lists") or [])
        stamped: list[tuple[int, SeriesHit]] = []
        seen: set[int] = set()
        for lst in lists:
            for entry in (lst or {}).get("entries") or []:
                entry = entry or {}
                m = entry.get("media") or {}
                mid = m.get("id")
                if mid is None or mid in seen:
                    continue
                seen.add(mid)
                updated = entry.get("updatedAt")
                stamped.append((updated if isinstance(updated, int) else 0,
                                _hit_from_media(m)))
        stamped.sort(key=lambda pair: pair[0], reverse=True)
        return [hit for _updated, hit in stamped]
