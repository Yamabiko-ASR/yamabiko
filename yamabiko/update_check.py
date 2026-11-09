"""Startup update check against GitHub Releases.

Best-effort, non-blocking. Silently skips on network failure, rate
limits, or malformed responses. No telemetry -- just one unauthenticated
GET to the public GitHub API.

Lists releases instead of hitting /releases/latest: that endpoint
EXCLUDES prerelease-flagged releases, and the beta channel ships
prerelease-flagged tags -- /releases/latest would go dark for every
beta install. Prereleases are deliberately considered candidates here;
the beta channel IS the product right now.
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Optional, Tuple

import httpx
from packaging.version import InvalidVersion, Version

log = logging.getLogger(__name__)

_RELEASES_URL = "https://api.github.com/repos/Yamabiko-ASR/yamabiko/releases?per_page=10"


def _release_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    url = value.strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return url


def check_for_update(current_version: str) -> Optional[Tuple[str, str]]:
    """Return (latest_version, release_url) if an update is available, else None.

    Blocks on the HTTP call (~100-500 ms). Caller should run this off
    the main thread.
    """

    # Every outcome logs one INFO line (available / up-to-date / failed):
    # "why did I never hear about vNext?" must be answerable from a
    # default-level main-overlay.log (logging audit 2026-07-01).
    try:
        resp = httpx.get(
            _RELEASES_URL,
            timeout=5.0,
            follow_redirects=True,
            headers={"Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.info("update check failed: %s", exc)
        return None

    releases = data if isinstance(data, list) else []
    # Highest parseable non-draft version wins. Drafts are filtered
    # defensively (the anonymous API doesn't return them, but an
    # authenticated proxy might); prereleases are NOT filtered -- see
    # module docstring. Unparseable tags (e.g. 'nightly-build') are
    # skipped rather than fatal so one stray tag can't kill the check.
    best: Optional[Tuple[Version, str, str]] = None
    for release in releases:
        if not isinstance(release, dict) or release.get("draft"):
            continue
        tag = str(release.get("tag_name") or "")
        remote = tag.lstrip("v")
        try:
            version = Version(remote)
        except InvalidVersion:
            continue
        url = _release_url(release.get("html_url"))
        if url is None:
            continue
        if best is None or version > best[0]:
            best = (version, remote, url)

    if best is None:
        if isinstance(data, list) and not releases:
            # A repo with zero published releases is a legitimate
            # pre-first-release state, not a failure; the "failed"
            # wording must stay off this branch.
            log.info("no releases published yet; nothing to compare against")
        else:
            log.info(
                "update check failed: no parseable release among %d entries",
                len(releases),
            )
        return None

    version, remote, url = best
    try:
        current = Version(current_version)
    except InvalidVersion:
        log.info(
            "update check failed: unparseable current version %r",
            current_version,
        )
        return None
    if version > current:
        log.info("update available (%s -> %s)", current_version, remote)
        return (remote, url)
    log.info("up to date (%s)", current_version)
    return None
