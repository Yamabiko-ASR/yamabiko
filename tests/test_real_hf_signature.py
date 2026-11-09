"""Real-path check: huggingface_hub's hf_hub_download signature.

These tests are not mock-heavy by design. They exist because our
downloader depends on real signature details that mocked tests
would happily paper over. Two guards:

1. ``test_hf_hub_download_accepts_our_required_kwargs`` -- if
   ``repo_id`` / ``filename`` / ``revision`` / ``local_dir`` ever
   disappear, the wizard's download path breaks before any mocked
   test would notice.
2. ``test_installed_hub_delivers_progress_through_downloads_hook`` --
   our downloader injects byte progress version-tolerantly: via the
   ``tqdm_class=`` kwarg on huggingface_hub >= 1.1.0 (the bundle pin),
   or by rebinding ``huggingface_hub.utils.tqdm``'s module-global
   ``tqdm`` class on hubs that lack the kwarg (0.x / 1.0.x -- the
   local dev venv drifted to 0.36.2 once already, see pyproject.toml).
   This test drives whichever seam the INSTALLED hub exposes and
   fails if the hook stops delivering byte ticks.
"""
from __future__ import annotations

import inspect

import pytest


# Both tests below are offline/deterministic, but they live in the
# real gate: preflight section 13 treats every test_real_*.py file as
# fully real-tier, and the CI real-tier job's inventory names the
# hub-signature checks as part of `pytest -m real` -- an auditor of the
# real gate must actually see them run there, so the marker is explicit.
@pytest.mark.real
def test_hf_hub_download_accepts_our_required_kwargs() -> None:
    """Our code calls hf_hub_download with repo_id, filename, revision, local_dir.

    If any of these disappear from the public signature, the wizard's
    download path breaks before any mocked test would notice.
    """
    huggingface_hub = pytest.importorskip("huggingface_hub")
    sig = inspect.signature(huggingface_hub.hf_hub_download)
    params = set(sig.parameters)
    required = {"repo_id", "filename", "revision", "local_dir"}
    missing = required - params
    assert not missing, (
        f"hf_hub_download missing kwargs we rely on: {missing}; "
        f"downloads.py::_HFDownloader needs updating"
    )


@pytest.mark.real
def test_installed_hub_delivers_progress_through_downloads_hook() -> None:
    """Whichever progress mechanism the INSTALLED huggingface_hub
    exposes, the production hook in downloads.py must deliver byte
    ticks through it:

    - hub >= 1.1.0 (the bundle pin): ``hf_hub_download`` accepts
      ``tqdm_class=`` (added upstream in PR #3531). The hook must put
      a callback-emitting class into the call kwargs, and that class
      must tick even on a DISABLED bar (the frozen wizard has no TTY,
      and hub builds transfer bars with tqdm's disable-on-non-TTY
      semantics).
    - hub 0.x / 1.0.x (no ``tqdm_class`` -- e.g. the 0.36.2 the dev
      venv drifted to): the hook must rebind the module-global ``tqdm``
      class in ``huggingface_hub.utils.tqdm``, which BOTH real transfer
      paths (``http_get`` and ``xet_get``) instantiate at call time via
      ``_get_progress_bar_context``. NOTE: patching ``tqdm.auto.tqdm``
      (the pre-2026-04-23 approach) does NOT intercept this seam --
      hub's class is derived from it at import time (verified by probe
      2026-07-10).

    No mocks on the subject: the fallback branch drives hub's real
    ``_get_progress_bar_context`` (no network involved).

    Seeded one-line defects that turn this red:
    - fallback branch: delete the ``module.tqdm = patched`` rebinding
      in downloads.py's ``_hub_progress_hook`` -> no byte ticks.
    - tqdm_class branch: delete the ``kwargs["tqdm_class"] = ...``
      line -> the injected-kwarg assert fails.
    - either branch: revert ``_make_progress_tqdm`` to reading
      ``self.n`` (a disabled bar never increments it) -> ticks report
      0 bytes done.
    """
    huggingface_hub = pytest.importorskip("huggingface_hub")
    import importlib
    import logging

    from yamabiko.wizard_pages.downloads import _hub_progress_hook

    hub_tqdm_mod = importlib.import_module("huggingface_hub.utils.tqdm")
    original_cls = hub_tqdm_mod.tqdm

    sig = inspect.signature(huggingface_hub.hf_hub_download)
    received: list = []
    kwargs: dict = {}
    with _hub_progress_hook(
        lambda done, total: received.append((done, total)), kwargs,
    ):
        if "tqdm_class" in sig.parameters:
            # hub >= 1.1.0: the hook injects the kwarg; drive the
            # injected class the way hub drives a transfer bar on a
            # non-TTY process (disabled bar).
            assert "tqdm_class" in kwargs, (
                "hub accepts tqdm_class but the hook did not inject it"
            )
            bar = kwargs["tqdm_class"](total=4096, disable=True, unit="B")
            try:
                bar.update(1024)
            finally:
                bar.close()
        else:
            # hub 0.x / 1.0.x: the kwarg would TypeError; the hook must
            # NOT inject it and must instead intercept the real seam.
            assert "tqdm_class" not in kwargs, (
                "hook injected tqdm_class into a hub whose "
                "hf_hub_download rejects it (TypeError on first "
                "wizard download)"
            )
            with hub_tqdm_mod._get_progress_bar_context(
                desc="canary", log_level=logging.INFO, total=4096,
            ) as bar:
                bar.update(1024)
    assert received == [(1024, 4096)], (
        f"wizard byte-progress is disconnected on the installed "
        f"huggingface_hub {huggingface_hub.__version__}: expected one "
        f"(1024, 4096) tick, got {received!r}"
    )
    assert hub_tqdm_mod.tqdm is original_cls, (
        "_hub_progress_hook leaked its patched class into "
        "huggingface_hub.utils.tqdm after the download finished"
    )


# ---------------------------------------------------------------------------
# Regression guard for HF repo coverage.
#
# A 401 once shipped because an invented repo string was
# never verified against the live HF API. The wizard SmokePage only
# touches one repo; that unit test isn't enough. Walk every repo our
# wizard's DownloadsPage enqueues and verify each exists + exposes
# the exact filename we request, AT the pinned revision the queue's
# DownloadSpecs carry -- upstream `main` is a snapshot no fresh
# install ever downloads.
# ---------------------------------------------------------------------------

import pytest as _pytest_real
_real = _pytest_real.mark.real


@_real
def test_all_wizard_hf_repos_exist_with_expected_files() -> None:
    """Every (repo_id, revision, filename) triple the wizard enqueues must
    resolve via huggingface_hub.list_repo_files AT THE PINNED REVISION --
    the wizard's DownloadSpec queue carries a commit sha for every item
    (downloads.py::download_queue), so a check against upstream
    `main` verifies a snapshot no fresh install ever downloads. A
    transposed hex char in a pin, or a pin predating a file's upload
    (e.g. a translator-repo commit before the small-tier quant landed),
    ships green under a main-ref check and 404s every fresh install.

    Source of truth is the constants block in
    yamabiko/wizard_pages/downloads.py (re-exported from yamabiko.models).
    Import them rather than hardcoding here so a rename in one place
    doesn't silently desync two places.

    The exact byte size of every enqueued file must equal
    ``models.pinned_size`` at that revision: the wizard's overall "X of Y GB",
    its time left and its row sizes come from those pins (L4).

    Seeded defects (each one line): transpose two hex chars in a
    translator pin's revision -> list_repo_files raises
    RevisionNotFoundError (404, an HfHubHTTPError) -> red; repoint a pin
    at an early upstream commit that predates its quant's upload -> that
    tier's filename assertion goes red. The old unpinned body caught
    neither. Change one digit of a size in ``models._PINNED_SIZES`` -> the
    size assertion goes red.
    """
    from yamabiko.models import pinned_size

    huggingface_hub = _pytest_real.importorskip("huggingface_hub")
    httpx = _pytest_real.importorskip("httpx")
    from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled
    try:
        from yamabiko.wizard_pages.downloads import (
            _TRANSLATOR_PINS,
            _WHISPER_REPO, _WHISPER_FILES, _WHISPER_REVISION,
            _SILERO_REPO, _SILERO_FILE, _SILERO_REVISION,
        )
    except ImportError as exc:
        _pytest_real.skip(f"downloads.py constants unavailable: {exc}")

    # One list_repo_files call per (repo, revision) snapshot, then every
    # enqueued filename is asserted against that listing. This mirrors
    # the wizard queue exactly: every translator quant tier (the path
    # parameterised by user choice), all five whisper CT2 files, and the
    # silero VAD file, each at the same pinned revision the DownloadSpec
    # carries. Tier pins sharing a (repo, revision) merge into one call.
    tier_checks: dict[tuple, list] = {}
    for pin in _TRANSLATOR_PINS.values():
        tier_checks.setdefault((pin.repo, pin.revision), []).append(pin.filename)
    checks = [
        *((repo, rev, files) for (repo, rev), files in tier_checks.items()),
        (_WHISPER_REPO, _WHISPER_REVISION, list(_WHISPER_FILES)),
        (_SILERO_REPO, _SILERO_REVISION, [_SILERO_FILE]),
    ]
    for repo_id, revision, filenames in checks:
        # Skip ONLY for environment problems (no network, offline mode,
        # rate limit) -- same narrow idiom as test_downloads.py's
        # _download_or_skip. A 401/404 (RepositoryNotFoundError /
        # RevisionNotFoundError / GatedRepoError, all HfHubHTTPError
        # subclasses) IS the defect this test exists to catch and must
        # FAIL, not skip: the old blanket `except Exception -> skip` kept
        # the suite green while every fresh install's first download died
        # (the invented-repo bug class routed around its own tripwire).
        try:
            files = huggingface_hub.list_repo_files(repo_id, revision=revision)
            infos = huggingface_hub.HfApi().get_paths_info(
                repo_id, list(filenames), revision=revision,
            )
        except (httpx.TransportError, OfflineModeIsEnabled) as exc:
            _pytest_real.skip(f"HF API unreachable for {repo_id}: {exc}")
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429:
                _pytest_real.skip(f"HF API rate-limited for {repo_id}: {exc}")
            raise
        for filename in filenames:
            assert filename in files, (
                f"repo {repo_id!r} at pinned revision {revision!r} does not "
                f"expose {filename!r}; wizard download would 404 on every "
                f"fresh install. Available: {files[:10]}..."
            )
        sizes = {info.path: info.size for info in infos}
        for filename in filenames:
            pinned = pinned_size(repo_id, filename, revision)
            assert sizes.get(filename) == pinned, (
                f"{repo_id}@{revision} {filename} is {sizes.get(filename)} bytes on "
                f"the Hub but models.pinned_size says {pinned}: the wizard's total, "
                f"time left and row sizes would be wrong"
            )
