"""Tests for yamabiko.update_check."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

from yamabiko.update_check import check_for_update


def _release(tag_name: str, html_url: str = "https://example.com/rel", *,
             draft: bool = False, prerelease: bool = False) -> dict:
    """One entry of the GET /releases list payload."""
    return {
        "tag_name": tag_name,
        "html_url": html_url,
        "draft": draft,
        "prerelease": prerelease,
    }


def _mock_response(*releases: dict):
    """Build a mock httpx.Response whose JSON body is a release LIST
    (the shape of GET /repos/.../releases, which the checker uses because
    /releases/latest excludes prerelease-flagged releases)."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = list(releases)
    return resp


class TestCheckForUpdate:
    @patch("httpx.get")
    def test_returns_newer_version(self, mock_get):
        mock_get.return_value = _mock_response(
            _release("v0.2.0-beta", "https://example.com/release"),
        )
        result = check_for_update("0.1.0-beta")
        assert result == ("0.2.0-beta", "https://example.com/release")

    @patch("httpx.get")
    def test_returns_none_when_current(self, mock_get):
        mock_get.return_value = _mock_response(_release("v0.1.0-beta"))
        result = check_for_update("0.1.0-beta")
        assert result is None

    @patch("httpx.get")
    def test_returns_none_when_older(self, mock_get):
        mock_get.return_value = _mock_response(_release("v0.0.9"))
        result = check_for_update("0.1.0-beta")
        assert result is None

    @patch("httpx.get")
    def test_prerelease_flagged_release_is_still_found(self, mock_get):
        """THE beta-channel case: v0.2.0-beta published with the GitHub
        "pre-release" checkbox ticked. /releases/latest EXCLUDES such
        releases, so a checker built on it goes silently dark for every
        beta install. The checker must list releases and consider
        prereleases. Seeded defect: skip entries with
        release.get("prerelease") -> red."""
        mock_get.return_value = _mock_response(
            _release("v0.2.0-beta", "https://example.com/beta", prerelease=True),
            _release("v0.1.0", "https://example.com/old"),
        )
        result = check_for_update("0.1.0")
        assert result == ("0.2.0-beta", "https://example.com/beta")

    @patch("httpx.get")
    def test_draft_release_is_excluded(self, mock_get):
        """An unpublished draft (visible to the API only with auth, but
        defensively filtered anyway) must never be announced as an
        update. Seeded defect: drop the draft filter -> the v0.3.0 draft
        wins and this goes red."""
        mock_get.return_value = _mock_response(
            _release("v0.3.0", "https://example.com/draft", draft=True),
            _release("v0.2.0-beta", "https://example.com/beta", prerelease=True),
        )
        result = check_for_update("0.1.0")
        assert result == ("0.2.0-beta", "https://example.com/beta")

    @patch("httpx.get")
    def test_unparseable_tag_is_skipped_not_fatal(self, mock_get):
        """A stray non-version tag (e.g. 'nightly-build') in the release
        list must not kill the whole check; parseable siblings still
        win. Seeded defect: return None on the first InvalidVersion ->
        red."""
        mock_get.return_value = _mock_response(
            _release("nightly-build", "https://example.com/nightly"),
            _release("v0.2.0-beta", "https://example.com/beta"),
        )
        result = check_for_update("0.1.0")
        assert result == ("0.2.0-beta", "https://example.com/beta")

    @patch("httpx.get")
    def test_picks_highest_version_not_list_order(self, mock_get):
        """The maximum parsed version wins, regardless of where GitHub
        puts it in the list. Seeded defect: take the first list entry ->
        red."""
        mock_get.return_value = _mock_response(
            _release("v0.1.5", "https://example.com/old"),
            _release("v0.2.0-beta", "https://example.com/new", prerelease=True),
        )
        result = check_for_update("0.1.0")
        assert result == ("0.2.0-beta", "https://example.com/new")

    @patch("httpx.get")
    def test_newer_release_without_html_url_is_ignored(self, mock_get):
        mock_get.return_value = _mock_response(
            _release("v9.0.0", html_url=""),
        )
        assert check_for_update("1.0.0") is None

    @patch("httpx.get")
    def test_invalid_higher_release_does_not_shadow_complete_lower_release(
        self, mock_get,
    ):
        mock_get.return_value = _mock_response(
            _release("v3.0.0", html_url="javascript:alert(1)"),
            _release(
                "v2.0.0",
                html_url="https://github.com/x/y/releases/tag/v2.0.0",
            ),
        )
        assert check_for_update("1.0.0") == (
            "2.0.0",
            "https://github.com/x/y/releases/tag/v2.0.0",
        )

    @patch("yamabiko.update_check.httpx.get", side_effect=Exception("network down"))
    def test_returns_none_on_network_error(self, mock_get):
        result = check_for_update("0.1.0-beta")
        assert result is None

    @patch("httpx.get")
    def test_returns_none_on_malformed_json(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"some_other_key": "value"}  # not a release list
        mock_get.return_value = resp
        result = check_for_update("0.1.0-beta")
        assert result is None

    @patch("httpx.get")
    def test_returns_none_when_no_release_has_parseable_tag(self, mock_get):
        mock_get.return_value = _mock_response(_release("not-a-version"))
        result = check_for_update("0.1.0-beta")
        assert result is None

    @patch("httpx.get")
    def test_empty_release_list_logs_no_releases_yet_without_failed_wording(
        self, mock_get, caplog,
    ):
        """A repo with zero published releases is a legitimate
        pre-first-release state, not a failure: the INFO line must say
        'no releases published yet' and must not carry the 'failed'
        wording ('update check failed: no parseable release among 0
        entries' reads as breakage in every pre-release install's log).

        Seeded defect: route the empty list through the
        no-parseable-release failure line -> 'failed' appears -> red.
        """
        mock_get.return_value = _mock_response()
        with caplog.at_level(logging.INFO, logger="yamabiko.update_check"):
            result = check_for_update("0.1.0-beta")
        assert result is None
        msgs = [r.getMessage() for r in caplog.records]
        assert not any("failed" in m for m in msgs), (
            f"an empty release list must not log as a failure: {msgs!r}"
        )
        assert any(
            "no releases published yet" in r.getMessage()
            and r.levelno == logging.INFO
            for r in caplog.records
        ), f"records: {msgs!r}"

    @patch("httpx.get")
    def test_http_error_returns_none_and_logs_at_info(self, mock_get, caplog):
        """A non-2xx status (rate limit, repo gone private) must produce
        None plus the single INFO 'update check failed' line -- the
        logging contract's failed outcome. Seeded defect: drop
        raise_for_status (treat the error body as a release list) ->
        no failure log, red."""
        resp = MagicMock()
        resp.status_code = 403
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403 rate limited", request=MagicMock(), response=MagicMock(),
        )
        mock_get.return_value = resp
        with caplog.at_level(logging.INFO, logger="yamabiko.update_check"):
            result = check_for_update("0.1.0-beta")
        assert result is None
        assert any(
            "update check failed" in r.getMessage() and r.levelno == logging.INFO
            for r in caplog.records
        ), f"records: {[r.getMessage() for r in caplog.records]!r}"


class TestConfigToggle:
    """Verify that the overlay_app wiring respects updates.check_on_startup."""

    @patch("yamabiko.overlay_app.__main__.load_full_config", return_value={"updates": {"check_on_startup": False}})
    @patch("yamabiko.overlay_app.__main__.check_for_update")
    def test_config_false_skips_check(self, mock_check, mock_cfg):
        from yamabiko.overlay_app.__main__ import _schedule_update_check
        result = _schedule_update_check(MagicMock())
        assert result is None
        mock_check.assert_not_called()


# ---------------------------------------------------------------------------
# The ENABLED path -- _on_update wiring + thread start must be exercised
# with invocation tests: deleting the
# `thread.updateAvailable.connect(_on_update)` or `thread.start()` line in
# overlay_app/__main__.py silently kills update notifications forever while
# the suite stays green, because both main()-harnesses stub
# _schedule_update_check.
# ---------------------------------------------------------------------------


def _enabled_cfg(monkeypatch, overlay_main):
    monkeypatch.setattr(
        overlay_main, "load_full_config",
        lambda: {"updates": {"check_on_startup": True}},
    )


def test_schedule_update_check_enabled_starts_thread_and_shows_tray_message(
    qtbot, monkeypatch,
):
    """Enabled config: _schedule_update_check must construct the REAL
    _UpdateCheckThread, connect updateAvailable to the notification
    closure, START the thread, and return it (caller keeps the ref so
    it isn't GC'd mid-flight). End-to-end observable effect: the tray's
    show_toast fires with the new version string.

    Seeded-defect drill: delete `thread.updateAvailable.connect(_on_update)`
    -> the thread runs but show_toast never fires -> waitUntil times out
    -> red. Delete `thread.start()` -> same -> red.
    """
    import yamabiko.overlay_app.__main__ as overlay_main

    _enabled_cfg(monkeypatch, overlay_main)
    monkeypatch.setattr(
        overlay_main, "check_for_update",
        lambda current: ("9.9.9", "https://github.com/example/rel"),
    )
    tray = MagicMock()

    thread = overlay_main._schedule_update_check(tray)
    try:
        assert thread is not None, (
            "_schedule_update_check returned None on the enabled path"
        )
        qtbot.waitUntil(lambda: tray.show_toast.called, timeout=5000)
    finally:
        if thread is not None:
            assert thread.wait(5000), "update-check thread did not finish"

    args = tray.show_toast.call_args.args
    assert any("9.9.9" in str(a) for a in args), (
        f"notification does not carry the new version string: {args}"
    )


def test_update_toast_click_opens_release_url_outside_job(qtbot, monkeypatch):
    """The update toast's click handler must open the release URL through
    open_url_outside_job (so the browser is not a member of the overlay's
    Job Objects and survives a pipeline restart or Quit).

    Seeded-defect drill: drop `on_click=` from the show_toast call in
    _on_update -> no click handler -> red. Open the URL with
    QDesktopServices again -> open_url_outside_job never sees it -> red.
    """
    import yamabiko.overlay_app.__main__ as overlay_main

    _enabled_cfg(monkeypatch, overlay_main)
    url = "https://github.com/example/release-v9"
    monkeypatch.setattr(
        overlay_main, "check_for_update", lambda current: ("9.9.9", url),
    )
    opened: list[str] = []
    monkeypatch.setattr(overlay_main, "open_url_outside_job", opened.append)
    tray = MagicMock()

    thread = overlay_main._schedule_update_check(tray)
    try:
        qtbot.waitUntil(lambda: tray.show_toast.called, timeout=5000)
    finally:
        if thread is not None:
            assert thread.wait(5000)

    click_handler = tray.show_toast.call_args.kwargs.get("on_click")
    assert callable(click_handler), (
        "_on_update did not give the update toast a click handler"
    )
    click_handler()  # simulate the user clicking the balloon
    assert opened == [url], f"click did not open the release page: {opened}"


def test_update_thread_up_to_date_result_shows_no_notification(
    qtbot, monkeypatch,
):
    """check_for_update returning None (already up to date / network down)
    must produce NO tray notification — the run body's
    `if result is not None` guard.

    Seeded-defect drill: change _UpdateCheckThread.run to
    `result = check_for_update(...) or ("0.0.0", "")` (notify even when
    up to date) -> show_toast fires -> red.
    """
    import yamabiko.overlay_app.__main__ as overlay_main

    _enabled_cfg(monkeypatch, overlay_main)
    monkeypatch.setattr(overlay_main, "check_for_update", lambda current: None)
    tray = MagicMock()

    thread = overlay_main._schedule_update_check(tray)
    assert thread is not None
    assert thread.wait(5000), "update-check thread did not finish"
    # Drain any queued deliveries before asserting the negative.
    qtbot.wait(100)
    assert not tray.show_toast.called, (
        "tray notification fired although no update is available"
    )


# ---------------------------------------------------------------------------
# Update-check visibility at INFO (logging-hardening Task 5): every
# outcome -- available / up-to-date / failed / disabled -- must leave one
# INFO line, so "why did I never hear about vNext?" is answerable from a
# default-level log. caplog.at_level(INFO) drops DEBUG records entirely,
# so demoting any of these lines back to log.debug goes red.
# ---------------------------------------------------------------------------


def test_check_for_update_logs_update_available_at_info(caplog):
    """Seeded defect: delete the 'update available' log.info line (or
    demote it to debug) -> no matching INFO record, red."""
    with patch(
        "yamabiko.update_check.httpx.get",
        return_value=_mock_response(
            _release("v0.2.0-beta", "https://example.com/rel"),
        ),
    ):
        with caplog.at_level(logging.INFO, logger="yamabiko.update_check"):
            result = check_for_update("0.1.0-beta")
    assert result is not None
    assert any(
        "update available (0.1.0-beta -> 0.2.0-beta)" in r.getMessage()
        and r.levelno == logging.INFO
        for r in caplog.records
    ), f"records: {[r.getMessage() for r in caplog.records]!r}"


def test_check_for_update_logs_up_to_date_at_info(caplog):
    """Seeded defect: delete the 'up to date' log.info line -> red."""
    with patch(
        "yamabiko.update_check.httpx.get",
        return_value=_mock_response(_release("v0.1.0-beta")),
    ):
        with caplog.at_level(logging.INFO, logger="yamabiko.update_check"):
            result = check_for_update("0.1.0-beta")
    assert result is None
    assert any(
        "up to date (0.1.0-beta)" in r.getMessage()
        and r.levelno == logging.INFO
        for r in caplog.records
    ), f"records: {[r.getMessage() for r in caplog.records]!r}"


def test_check_for_update_logs_check_failed_at_info(caplog):
    """Network failure was previously log.debug -- invisible at the
    default level, so a user permanently offline from api.github.com
    could never tell. Seeded defect: demote the failure line back to
    log.debug -> caplog.at_level(INFO) sees no record, red."""
    with patch(
        "yamabiko.update_check.httpx.get",
        side_effect=Exception("network down"),
    ):
        with caplog.at_level(logging.INFO, logger="yamabiko.update_check"):
            result = check_for_update("0.1.0-beta")
    assert result is None
    assert any(
        "update check failed" in r.getMessage()
        and "network down" in r.getMessage()
        and r.levelno == logging.INFO
        for r in caplog.records
    ), f"records: {[r.getMessage() for r in caplog.records]!r}"


def test_schedule_update_check_disabled_logs_at_info(monkeypatch, caplog):
    """The config-disabled outcome must also leave an INFO line (it was
    log.debug). Seeded defect: demote back to debug -> red."""
    import yamabiko.overlay_app.__main__ as overlay_main

    monkeypatch.setattr(
        overlay_main, "load_full_config",
        lambda: {"updates": {"check_on_startup": False}},
    )
    with caplog.at_level(logging.INFO):
        result = overlay_main._schedule_update_check(MagicMock())
    assert result is None
    assert any(
        "update check disabled by config" in r.getMessage()
        and r.levelno == logging.INFO
        for r in caplog.records
    ), f"records: {[r.getMessage() for r in caplog.records]!r}"


# ---------------------------------------------------------------------------
# Real-path integration — hit the actual GitHub API
# ---------------------------------------------------------------------------

@pytest.mark.real
def test_real_github_api_handles_bootstrap_or_returns_sensible_payload():
    """Actually hit GitHub's releases API for the Yamabiko repo.

    Before the first public release, the private-repo 404 and empty-release
    states must take production's quiet ``None`` path. Once releases exist,
    this catches schema drift or a broken release URL that mocked tests miss.

    We don't assert the returned version (it changes with every release).
    We DO assert the code path completes without exception and returns
    either None (up-to-date) or (version, url) where url points at github.com.
    """
    import httpx

    # Best-effort: skip if offline / rate-limited / GitHub is down.
    # We want this test to be informative when things actually break,
    # not flaky when the network is having a bad day. Probe the same
    # list endpoint production uses (NOT /releases/latest, which 404s
    # on a repo whose only releases are prerelease-flagged -- the exact
    # beta-channel case the checker exists for).
    try:
        probe = httpx.get(
            "https://api.github.com/repos/Yamabiko-ASR/yamabiko/releases?per_page=10",
            timeout=5.0,
            headers={"Accept": "application/vnd.github+json"},
        )
    except Exception as exc:
        pytest.skip(f"GitHub API unreachable: {exc!r}")

    if probe.status_code == 404:
        assert check_for_update("0.0.0-test") is None
        return
    if probe.status_code in (403, 429):
        pytest.skip(f"GitHub rate-limited ({probe.status_code}) — try again later")
    if probe.status_code != 200:
        pytest.skip(f"unexpected GitHub API status {probe.status_code}")
    if not probe.json():
        assert check_for_update("0.0.0-test") is None
        return

    # Now run the real check — it should hit the same endpoint cleanly.
    result = check_for_update("0.0.0-test")  # force remote > local path
    # Result is either None (shouldn't happen with 0.0.0-test baseline) or
    # a (version, url) tuple.
    assert result is not None, (
        "check_for_update returned None against 0.0.0-test baseline — "
        "schema drift on the real API response?"
    )
    version, url = result
    assert isinstance(version, str) and len(version) > 0
    assert url.startswith("https://github.com/"), f"unexpected URL: {url}"
