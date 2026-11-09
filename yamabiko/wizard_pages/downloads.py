"""Wizard step "Download": sequential model downloads with byte-level progress.

Pulls the translator GGUF, faster-whisper CT2, and Silero VAD from
HuggingFace.
Files land in HF's cache
at ``HF_HOME`` — which ``yamabiko.hf_env`` pins to
``%LOCALAPPDATA%\\Yamabiko\\hf-cache\\``. The wizard and the runtime
pipeline both use plain ``hf_hub_download()``, so once the wizard has
warmed the cache, every runtime lookup is an instant local read.

Design principles:

- **Delegate transfer to huggingface_hub.** ETag checks, xet chunk-based
  transfer, automatic retry on transient errors are all built in.
  Previous code reimplemented these badly; the rewrite trusts the
  library. Interrupted downloads are NOT resumed: a cancelled or killed
  run leaves a ``blobs/*.incomplete`` file that the xet path cannot
  continue, so ``_sweep_incomplete_blobs`` deletes those for the queued
  repos before each queue starts and the file is fetched again.
- **Pinned digests checked on landing.** Files with a SHA-256 pin in
  ``yamabiko.models`` (the whisper ``model.bin``, the translator GGUFs, Silero)
  are hashed after ``hf_hub_download`` returns; a mismatch deletes the
  file and fails that item.
- **No ``local_dir`` flat-layout hack.** HF's cache layout
  (``hub/models--<org>--<name>/snapshots/<sha>/...``) is what the runtime
  expects, because the runtime also uses plain ``hf_hub_download``. No
  subdir-flatten move and no own skip-check.
- **Progress via a version-tolerant hook.** HF doesn't expose a
  progress callback. On huggingface_hub >= 1.1.0 (the bundle pin)
  ``hf_hub_download`` accepts ``tqdm_class=`` (added upstream in
  PR #3531; 1.0.0 / 1.0.1 and all of 0.x lack it) and we pass a
  subclass that emits on each ``.update(n)``. On a hub without the
  kwarg (a drifted dev venv landed on 0.36.2 once — see
  pyproject.toml) passing it raises ``TypeError`` on the first
  wizard download, so ``_hub_progress_hook`` instead rebinds the
  module-global ``tqdm`` class in ``huggingface_hub.utils.tqdm``
  for the duration of the call — the class BOTH 0.x transfer paths
  (``http_get`` and ``xet_get``) instantiate at call time. Note:
  monkeypatching ``tqdm.auto.tqdm`` (this file's pre-2026-04-23
  approach) does NOT intercept that seam — hub derives its own
  class from it at import time (probed 2026-07-10). Guard test
  ``test_installed_hub_delivers_progress_through_downloads_hook``
  fails if the installed hub stops supporting whichever mechanism
  the hook selects.
- **Retry in the step's alert** on failure. HF's own retry loop has
  already exhausted itself -- Retry re-runs the queue, and the
  already-downloaded files no-op from the cache (after their digest check).
  Leaving the step never cancels; only closing the wizard does
  (``shutdown_thread``).

The ``QThread`` worker keeps the Qt main thread responsive during the
multi-GB translator download.
"""
from __future__ import annotations

import contextlib
import datetime
import errno
import hashlib
import importlib
import inspect
import logging
import math
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Protocol

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget, QWizardPage,
)

# Side-effect import: pins HF_HOME + HF_XET_HIGH_PERFORMANCE before any
# huggingface_hub code runs. Must stay at module top.
from yamabiko import hf_env  # noqa: F401

from yamabiko import config as _config
from yamabiko import paths
from yamabiko.overlay_app.controls import ProgressLine, ui_font
from yamabiko.overlay_app.theme import MIST
from yamabiko.ui import thread_keeper
from yamabiko.ui.error_report import redact_user
from yamabiko.wizard_pages.parts import Alert, Run, StatusRows, set_primary

_log = logging.getLogger(__name__)

# How often to log a progress sample during a long download. Every chunk
# would spam the log; once every few seconds is useful forensic data if
# a download stalls.
_PROGRESS_LOG_INTERVAL_SECONDS = 5.0

# Serializes _hub_progress_hook's hub-0.x fallback branch. That branch
# rebinds the PROCESS-GLOBAL ``huggingface_hub.utils.tqdm.tqdm`` class,
# so two overlapping downloads (a cancelled worker detached into
# thread_keeper and still draining, plus a fresh Retry / re-Apply
# worker) would otherwise chain their patches: the second worker
# captures the first's patched class as "original", and every tick of
# the second (healthy) download then runs the first (cancelled)
# worker's callback -- which raises _DownloadCancelledError inside the
# healthy download's stack. Held for the whole patch-yield-restore
# span, so overlapping fallback downloads block each other instead of
# corrupting the global. Only the drifted-dev-env fallback pays this
# cost; the hub >= 1.1.0 ``tqdm_class=`` branch is per-call state and
# stays lock-free.
_HUB_TQDM_PATCH_LOCK = threading.Lock()

# ALL repo / file / revision pins come from the single source of truth
# (yamabiko.models) so the wizard's download list cannot drift from the
# runtime resolvers (watchdog.py, asr_client.py, silero_chunker.py)
# or the preflight gate -- a producer/consumer
# mismatch.
# The translator quant tiers each carry their own repo + file + revision
# (yamabiko.models.TRANSLATOR_PINS: default ~5.1 GB for 10 GB+ VRAM, small
# ~4 GB for 8 GB cards; the tiers may live in DIFFERENT repos), selected
# via `translation.quant_tier` -- see `_resolve_translator_pin`.
from yamabiko.models import (  # noqa: E402
    TRANSLATOR_PINS as _TRANSLATOR_PINS,
    TranslatorPin as _TranslatorPin,
    translator_pin as _translator_pin,
    WHISPER_REPO as _WHISPER_REPO,
    WHISPER_FILES as _WHISPER_FILES,
    WHISPER_REVISION as _WHISPER_REVISION,
    SILERO_REPO as _SILERO_REPO,
    SILERO_FILE as _SILERO_FILE,
    SILERO_REVISION as _SILERO_REVISION,
    ModelArtifactCorrupt as _ModelArtifactCorrupt,
    _repo_files_cached,
    pinned_sha256 as _pinned_sha256,
    pinned_size as _pinned_size,
)


def _resolve_translator_pin() -> _TranslatorPin:
    """Pick the translator (repo, filename, revision) pin for the active
    quant tier.

    Reads ``translation.quant_tier`` from the merged config (defaults +
    user overrides). Unknown values fall back to the default tier with
    a warning rather than crashing the wizard -- a user with a malformed
    ``user_overrides.yaml`` shouldn't be unable to install.
    """
    import logging

    try:
        cfg = _config.load_full_config()
    except Exception:
        # Config load failure is surfaced elsewhere — fall back to
        # default so this codepath can't make things worse.
        return _TRANSLATOR_PINS["default"]

    tier = (cfg.get("translation", {}) or {}).get("quant_tier", "default")
    if tier not in _TRANSLATOR_PINS:
        logging.getLogger(__name__).warning(
            "unknown translation.quant_tier=%r; using default", tier,
        )
    return _translator_pin(tier)


_DISK_FULL_MARKERS = ("errno 28", "no space left", "not enough space on the disk", "os error 112")


def _is_disk_full(exc: BaseException) -> bool:
    """True for an out-of-space error, raised directly (errno ENOSPC) or
    wrapped by the hub: Python's "[Errno 28] No space left on device" and
    the xet transfer's Rust text "There is not enough space on the disk.
    (os error 112)"."""
    if getattr(exc, "errno", None) == errno.ENOSPC:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _DISK_FULL_MARKERS)


def _friendly_error(exc: Exception) -> str:
    """Map raw download exceptions to human-readable messages.

    Classified by substring match on the exception text. The Download
    step's alert and Settings' download dialog also expose raw details via
    Copy details for users who want to file bug reports.
    """
    msg = str(exc)
    if "401" in msg:
        return (
            "Model repository not accessible (removed or authentication "
            "required). Please report this to the developer."
        )
    if "403" in msg:
        return (
            "Access denied to the model repository. It may require a "
            "HuggingFace account or token."
        )
    if "404" in msg:
        return (
            "Model file not found on the server. Please report this to "
            "the developer."
        )
    if "timeout" in msg.lower() or "timed out" in msg.lower():
        return "Connection timed out. Check your internet connection and try again."
    if "ConnectionError" in type(exc).__name__ or "connection" in msg.lower():
        return "Could not connect to the download server. Check your internet connection."
    if _is_disk_full(exc):
        return (
            "Not enough disk space. Free up space and try again. Note: a "
            "partially downloaded *.incomplete file in Yamabiko's hf-cache "
            "folder also counts toward the space to clear."
        )
    return f"Download error: {msg[:200]}"


class _DownloadCancelledError(Exception):
    """Raised (once) from the worker's progress callback on user cancel.

    Aborts the blocking ``hf_hub_download`` mid-file on the plain-HTTP
    path. The xet path never aborts mid-file: pre-1.19 huggingface_hub
    swallowed callback exceptions silently; hub 1.19's XetSession
    reports each one via sys.excepthook and continues the transfer,
    which is why the worker raises at most once per queue.
    Either way xet cancellation takes effect at the between-files
    check; the wizard-close path detaches the thread via
    ``yamabiko.ui.thread_keeper`` instead of waiting.
    """


@dataclass
class DownloadSpec:
    """One file to pull from HuggingFace.

    The landed on-disk path comes from ``hf_hub_download()``'s return
    value — it lives under ``HF_HOME/hub/models--<org>--<name>/snapshots/
    <sha>/<filename>``. We don't compute or predict it; HF owns the
    cache layout.

    ``revision=None`` uses the repo's default branch. Pin a revision
    only when reproducibility matters (e.g. a pinned commit sha).
    """
    repo_id: str
    filename: str
    revision: Optional[str] = None


def download_queue() -> List[DownloadSpec]:
    """Every file the Download step fetches, for the active quant tier."""
    translator = _resolve_translator_pin()
    # Rows fill top to bottom (L5): Speech recognition, Translation
    # model, Voice detection. Every spec carries its pinned revision so
    # the wizard warms the exact snapshot the runtime resolves offline
    # (asr/asr_client.py, watchdog._resolve_translator_model_path,
    # vad/silero_chunker.py).
    return [
        *(DownloadSpec(_WHISPER_REPO, fn, revision=_WHISPER_REVISION)
          for fn in _WHISPER_FILES),
        DownloadSpec(translator.repo, translator.filename,
                     revision=translator.revision),
        DownloadSpec(_SILERO_REPO, _SILERO_FILE, revision=_SILERO_REVISION),
    ]


def pending_download_bytes(specs: List[DownloadSpec]) -> int:
    """Pinned bytes of the ``specs`` not in the local HF cache yet. A cache
    probe error counts that spec as not cached, as in ``_space_check``."""
    total = 0
    for spec in specs:
        try:
            cached = _repo_files_cached(spec.repo_id, (spec.filename,), spec.revision)
        except Exception:
            cached = False
        if not cached:
            total += _pinned_size(spec.repo_id, spec.filename, spec.revision) or 0
    return total


# ---------------------------------------------------------------------------
# Disk-space check. The queue used to start blind: an ENOSPC only surfaced
# 20-60 minutes in, mid-GGUF, leaving a multi-GB *.incomplete blob in the HF
# cache. The check uses conservative per-file ceilings (a margin over the
# exact sizes pinned in yamabiko.models) plus a working room, and runs before
# every queue start and again after a mid-download ENOSPC.
# ---------------------------------------------------------------------------

_GIB = 1024 ** 3

# Ceilings in bytes, keyed by the pinned GGUF filenames so a tier swap in
# models.py automatically re-keys them.
_PREFLIGHT_GGUF_BYTES = {
    _TRANSLATOR_PINS["default"].filename: int(5.2 * _GIB),
    _TRANSLATOR_PINS["small"].filename: int(4.1 * _GIB),
}
_PREFLIGHT_WHISPER_MODEL_BYTES = int(1.6 * _GIB)  # CT2 model.bin
_PREFLIGHT_SMALL_FILE_BYTES = 16 * 1024 ** 2      # configs / tokenizer / VAD
# Headroom the cache volume must keep beyond the estimated queue: HF writes
# through a temp *.incomplete file, and a drive squeezed to zero mid-install
# breaks more than the download.
_PREFLIGHT_MARGIN_BYTES = 1 * _GIB


def _estimated_spec_bytes(spec: DownloadSpec) -> int:
    """Conservative size ceiling for one queue entry (bytes)."""
    if spec.filename in _PREFLIGHT_GGUF_BYTES:
        return _PREFLIGHT_GGUF_BYTES[spec.filename]
    if spec.filename.endswith(".gguf"):
        # Unknown translator quant (config drift): assume the biggest tier.
        return max(_PREFLIGHT_GGUF_BYTES.values())
    if spec.filename == "model.bin":
        return _PREFLIGHT_WHISPER_MODEL_BYTES
    return _PREFLIGHT_SMALL_FILE_BYTES


def _space_shortfall(
    required_bytes: int,
    free_bytes: int,
    margin_bytes: int = _PREFLIGHT_MARGIN_BYTES,
) -> int:
    """Bytes to free before the queue may start: 0 when ``free_bytes``
    holds the uncached queue plus the working room, or nothing is needed.

    Pure decision logic (no disk, no Qt), so the threshold is directly
    unit-testable.
    """
    if required_bytes <= 0:
        return 0
    return max(0, required_bytes + margin_bytes - free_bytes)


@dataclass(frozen=True)
class _SpaceCheck:
    """The disk picture for a queue: what its uncached files need, what the
    cache volume has free, which drive that is, and the cache folder."""
    required: int
    free: int
    drive: str          # "C:", or "" when the cache is not on a drive letter
    cache_root: Path

    @property
    def shortfall(self) -> int:
        return _space_shortfall(self.required, self.free)


def _drive_letter(path: Path) -> str:
    """"C:" for a path on a drive letter, junctions followed (a cache linked
    to another volume names that volume); "" otherwise (UNC paths)."""
    try:
        drive = path.resolve().drive
    except OSError:
        drive = path.drive
    return drive.upper() if re.fullmatch(r"[A-Za-z]:", drive) else ""


def _space_check(specs: List[DownloadSpec]) -> Optional[_SpaceCheck]:
    """Measure ``specs`` against the cache volume.

    Sums the conservative ceilings of the specs NOT already in the local HF
    cache (per-spec ``models._repo_files_cached`` probe -- already-landed
    files no-op via HF's ETag check and need no space) and reads
    ``shutil.disk_usage`` of the cache volume (``HF_HOME``, pinned by
    ``yamabiko.hf_env``; the deepest existing ancestor is probed because the
    cache dir may not exist yet on a true first run).

    Fails OPEN: any unexpected probe error logs and returns None -- a broken
    check must never block an install the downloads themselves might
    complete. A per-spec cache-probe error only counts that spec as still
    needed (the conservative direction for a space check).
    """
    try:
        required = 0
        for spec in specs:
            try:
                cached = _repo_files_cached(
                    spec.repo_id, (spec.filename,), spec.revision,
                )
            except Exception:
                cached = False
            if not cached:
                required += _estimated_spec_bytes(spec)
        cache_root = Path(
            os.environ.get("HF_HOME")
            or (paths.app_data_dir() / "hf-cache"),
        )
        probe = cache_root
        while not probe.exists():
            if probe.parent == probe:
                break
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        return _SpaceCheck(required, free, _drive_letter(probe), cache_root)
    except Exception:
        _log.warning(
            "disk-space check failed; starting the download queue anyway",
            exc_info=True,
        )
        return None


def _model_fetch():
    """Settings' amount / time-left helpers (format_amount,
    format_time_left, size_text, _TimeLeft), imported at call time:
    overlay_app.model_fetch imports this module at its top."""
    from yamabiko.overlay_app import model_fetch
    return model_fetch


def _disk_full_message(drive: str, shortfall: int) -> str:
    """The disk-full alert: "Not enough space on drive C:. Free up 2.1 GB,
    then click Retry." The amount is decimal GB rounded UP to 0.1 (freeing
    a rounded-down figure would leave the queue blocked)."""
    where = f"on drive {drive}" if drive else "on the drive that holds the models"
    if shortfall <= 0:
        return f"Not enough space {where}. Free up some space, then click Retry."
    amount = _model_fetch().size_text(math.ceil(shortfall / 1e8) / 10)
    return f"Not enough space {where}. Free up {amount}, then click Retry."


def _old_models_label(retired_bytes: int) -> str:
    """The delete button's text, "Delete old Yamabiko models (4.4 GB)": the
    old models' bytes in decimal GB to the nearest 0.1, and at least 0.1, so
    a small retired set (the 2 MB turn-detector alone) never reads "0 GB"
    (Decision 144)."""
    amount = _model_fetch().size_text(max(0.1, retired_bytes / 1e9))
    return f"Delete old Yamabiko models ({amount})"


def _disk_full_details(
    check: Optional[_SpaceCheck], retired_bytes: Optional[int], error: Optional[BaseException],
    *, legacy_bytes: int = 0,
) -> str:
    """Copy details for the disk-full state, with the user name removed.
    ``retired_bytes`` is None when the cache was not scanned (a shared
    HF_HOME, _hf_home_is_ours); ``legacy_bytes`` is the v0.1 model folders'
    size."""
    from yamabiko import __version__

    lines = [f"Yamabiko {__version__} download stopped: not enough disk space"]
    if check is not None:
        lines += [
            f"Drive: {check.drive or 'not a drive letter'}",
            f"Free: {check.free} bytes",
            f"Needed: {check.required} bytes plus {_PREFLIGHT_MARGIN_BYTES} bytes of working room",
            f"Model folder: {check.cache_root}",
        ]
    if retired_bytes is None:
        lines.append("Old Yamabiko models: not scanned (shared cache)")
    else:
        lines.append(f"Old Yamabiko models: {retired_bytes} bytes")
    lines.append(f"Old Yamabiko models (v0.1 folder): {legacy_bytes} bytes")
    if error is not None:
        lines.append(f"Error: {error!r}")
    lines.append(f"Time: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    return redact_user("\n".join(lines))


def _hf_home_is_ours() -> bool:
    """True iff ``HF_HOME`` is Yamabiko's own cache directory.

    ``yamabiko.hf_env`` only defaults HF_HOME, so a user who pointed it at a
    cache shared with other tools keeps that choice. Deleting "retired"
    repos from such a cache could remove models another tool still uses,
    so Delete old Yamabiko models is offered only for Yamabiko's own cache.
    """
    return os.path.normcase(os.environ.get("HF_HOME", "")) == os.path.normcase(
        str(paths.app_data_dir() / "hf-cache"),
    )


def _sweep_incomplete_blobs(
    repo_ids, cache_dir: Optional[Path] = None,
) -> "tuple[int, int]":
    """Delete ``blobs/*.incomplete`` for ``repo_ids``; return (count, bytes).

    A cancelled or killed download leaves its partial blob behind, and the
    xet transfer path does not resume it, so it only holds disk space the
    preflight then demands again. The single-instance mutex rules out a
    writer in another Yamabiko process; a cancelled worker of THIS process
    may still be draining into its partial file (``thread_keeper``), so the
    sweep is skipped while one runs. Never raises: a file that cannot be
    deleted is logged and left for the next sweep.
    """
    if thread_keeper.any_running():
        _log.info("partial download sweep skipped: a previous download is still draining")
        return 0, 0
    try:
        from huggingface_hub import constants as hub_constants
        from huggingface_hub.file_download import repo_folder_name

        root = Path(cache_dir) if cache_dir is not None else Path(hub_constants.HF_HUB_CACHE)
        count = 0
        total = 0
        for repo_id in sorted(set(repo_ids)):
            blobs = root / repo_folder_name(repo_id=repo_id, repo_type="model") / "blobs"
            for partial in blobs.glob("*.incomplete"):
                try:
                    size = partial.stat().st_size
                    partial.unlink()
                except OSError as exc:
                    _log.warning("could not delete partial download %s: %s", partial, exc)
                    continue
                count += 1
                total += size
    except Exception:
        _log.warning("partial download sweep failed", exc_info=True)
        return 0, 0
    _log.info("partial download sweep: deleted %d file(s), %d bytes", count, total)
    return count, total


def _file_sha256(path: Path) -> str:
    """Stream ``path`` through SHA-256 in 8 MiB chunks (multi-GB GGUFs)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Model repos retired from yamabiko.models by pin migrations (the shisa ->
# Qwen3.5 translator swap, the deepdml -> kotoba -> efwkjn -> Kodama-1 ASR
# moves, the turn-detector removal). Their multi-GB blobs stay behind in the
# shared HF cache on upgraded installs: nothing reads them again, but they
# count against the free space the preflight demands for the NEW pins.
# Deleted only through the Download step's "Delete old Yamabiko models"
# button when the disk is full. Must never overlap the live pins in
# yamabiko.models (guarded by a producer/consumer test) -- an overlap would
# delete a model the user just downloaded.
_RETIRED_MODEL_REPOS: frozenset = frozenset({
    "dahara1/shisa-v2.1-qwen3-8b-UD-japanese-imatrix",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "efwkjn/faster-whisper-ja-760M",
    "kotoba-tech/kotoba-whisper-v2.0-faster",
    "livekit/turn-detector",
})


def _scan_retired_cache_entries(
    cache_dir: Optional[Path] = None,
) -> "tuple[tuple[str, ...], int]":
    """Commit hashes + total on-disk bytes of retired-model revisions in
    the local HF cache.

    Local-only probe: ``scan_cache_dir`` walks the cache tree, no network.
    ``cache_dir=None`` scans the default cache (``HF_HOME``, pinned by
    ``yamabiko.hf_env``); the parameter exists so the real-tier test can
    drive the actual hub library against a fabricated cache. Fails soft to
    ``((), 0)``: the reclaim affordance is an escape hatch, and a scan
    failure (missing cache dir on a true first run, corrupted layout) must
    degrade to the plain block copy, never break the preflight UI.
    """
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir(cache_dir)
    except Exception:
        _log.warning(
            "HF cache scan for retired model entries failed", exc_info=True,
        )
        return (), 0
    revisions: list = []
    total = 0
    for repo in info.repos:
        if repo.repo_type != "model" or repo.repo_id not in _RETIRED_MODEL_REPOS:
            continue
        revisions.extend(rev.commit_hash for rev in repo.revisions)
        total += repo.size_on_disk
    return tuple(revisions), total


def _delete_retired_cache_revisions(
    revisions: "tuple[str, ...]", cache_dir: Optional[Path] = None,
) -> None:
    """Delete exactly ``revisions`` via the hub's delete strategy.

    ``delete_revisions`` removes each revision's snapshot, any blobs only
    those revisions reference, and the whole repo directory once its last
    revision goes; hashes no longer present are silently skipped, so a
    scan that went stale between the block and the click cannot raise.
    """
    from huggingface_hub import scan_cache_dir

    strategy = scan_cache_dir(cache_dir).delete_revisions(*revisions)
    _log.info(
        "freeing retired model cache entries: %d revision(s), ~%d bytes",
        len(revisions), strategy.expected_freed_size,
    )
    strategy.execute()


# The plain model folders v0.1 wrote under <app data>\models before the HF
# cache. Nothing reads them now; like the retired repos, they are deleted
# only through "Delete old Yamabiko models".
_V01_MODEL_DIRS: tuple[str, ...] = (
    "shisa", "faster-whisper-large-v3-turbo", "livekit-turn-detector", "silero",
)


def _legacy_models_root() -> Path:
    """The folder v0.1 kept its models in."""
    return paths.app_data_dir() / "models"


def _dir_size(path: Path) -> int:
    """Bytes of the files under ``path``. Never enters a symlink or a
    junction; an entry that can't be read counts as 0."""
    total = 0
    with os.scandir(path) as entries:
        for entry in entries:
            try:
                if entry.is_symlink() or entry.is_junction():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    total += _dir_size(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total


def _scan_legacy_model_dirs(
    root: Optional[Path] = None,
) -> "tuple[tuple[Path, ...], int]":
    """The v0.1 model folders under ``root`` (default: the app data's
    ``models``) and their total bytes. A folder that is a symlink or a
    junction is skipped, and a ``models`` folder that is one finds nothing:
    the delete must never reach through a link. Fails soft to ``((), 0)``,
    like ``_scan_retired_cache_entries``.
    """
    try:
        base = root if root is not None else _legacy_models_root()
        if base.is_symlink() or base.is_junction():
            return (), 0
        dirs = []
        for name in _V01_MODEL_DIRS:
            path = base / name
            if path.is_dir() and not (path.is_symlink() or path.is_junction()):
                dirs.append(path)
        total = sum(_dir_size(d) for d in dirs)
    except Exception:
        _log.warning("scan of the v0.1 model folders failed", exc_info=True)
        return (), 0
    return tuple(dirs), total


def _delete_legacy_model_dirs(dirs: "tuple[Path, ...]") -> None:
    """Delete ``dirs``, then their ``models`` folder if that left it empty.
    A folder already gone (deleted by hand while the alert was up, or by an
    earlier partial Delete) is skipped; any other delete error propagates.
    rmtree removes a junction inside a folder without entering it
    (Python 3.12)."""
    _log.info("freeing v0.1 model folders: %d folder(s)", len(dirs))
    for d in dirs:
        if os.path.lexists(d):
            shutil.rmtree(d)
    if dirs:
        with contextlib.suppress(OSError):  # anything else in it keeps it
            dirs[0].parent.rmdir()


def _make_progress_tqdm(callback: Callable[[int, int], None], base=None):
    """Return a tqdm subclass that calls ``callback(done, total)`` on each
    ``.update()``.

    ``base`` defaults to ``tqdm.auto.tqdm`` — what huggingface_hub >= 1.1.0
    documents for ``tqdm_class=``. The hub-0.x fallback in
    ``_hub_progress_hook`` passes hub's own ``huggingface_hub.utils.tqdm``
    class instead, so hub-specific ctor kwargs (``name=``) keep working.

    Byte counts are tracked on the instance rather than read straight off
    ``self.n``. On hub >= 1.1.0 hub builds this ``tqdm_class=`` bar without
    a ``disable`` argument (only its own ``huggingface_hub.utils.tqdm``
    subclasses get ``disable=None``, hub issue #4050), so tqdm's default
    ``disable=False`` keeps the bar live: ``self.n`` is authoritative (it
    includes the HTTP resume offset via ``initial=``), and the bar writes
    to ``sys.stderr`` from its constructor on (the frozen entry gives a
    missing stderr a sink, ``watchdog._guard_frozen_stdio``). On the hub-0.x
    fallback the bar is hub's own class with ``disable=None`` ("disable on
    non-TTY"), and a **disabled** tqdm never increments ``self.n``: when
    stderr is a pipe (not a TTY), ``self.n`` alone would freeze the byte
    readout at the resume offset. There we accumulate from that
    same initial value ourselves.
    """
    if base is None:
        from tqdm.auto import tqdm as base

    class _ProgressTqdm(base):
        def update(self, n=1):
            result = super().update(n)
            if getattr(self, "disable", False):
                self._yamabiko_done = (
                    getattr(self, "_yamabiko_done", self.n) + (n or 0)
                )
            else:
                self._yamabiko_done = self.n
            if self.total:
                callback(self._yamabiko_done, self.total)
            return result

    return _ProgressTqdm


@contextlib.contextmanager
def _hub_progress_hook(progress_callback: Callable[[int, int], None],
                       kwargs: dict):
    """Wire byte-level progress into an ``hf_hub_download(**kwargs)`` call,
    tolerating both huggingface_hub API generations.

    - hub >= 1.1.0 (the bundle pin in yamabiko/requirements.txt):
      ``hf_hub_download`` accepts ``tqdm_class=`` (upstream PR #3531) —
      inject a callback-emitting subclass into ``kwargs``.
    - hub 0.x / 1.0.x (no ``tqdm_class`` — e.g. the 0.36.2 a drifted dev
      venv resolved once, see pyproject.toml): passing the kwarg raises
      ``TypeError``. Instead, rebind the module-global ``tqdm`` class in
      ``huggingface_hub.utils.tqdm`` for the duration of the call — both
      transfer paths (``http_get`` and ``xet_get``) build their bar via
      ``_get_progress_bar_context``, which instantiates that global at
      call time. The subclass derives from hub's ORIGINAL class so
      hub-specific ctor kwargs (``name=``) keep working.

    The real constraint on the fallback: the rebound class is
    PROCESS-GLOBAL state, and downloads can overlap (a cancel detaches a
    still-draining worker via thread_keeper with a short join timeout;
    Retry / re-Apply starts a second worker while the first is still
    inside this hook). Unserialized, the second hook would capture the
    first's patched class as "original", chain its patch on top, and
    route every tick of the healthy new download into the cancelled
    worker's callback — which raises _DownloadCancelledError inside the
    healthy download's stack. So the entire patch-yield-restore span
    holds ``_HUB_TQDM_PATCH_LOCK``: an overlapping fallback download
    blocks until the earlier one restores, and restoration is
    unconditional (under the lock nothing else can have rebound the
    global). The ``tqdm_class=`` branch has no shared state and takes no
    lock.

    NOTE: ``huggingface_hub.utils.tqdm`` as an attribute resolves to the
    *class* (the utils package re-exports it); the module must come from
    the import system.
    """
    from huggingface_hub import hf_hub_download

    if "tqdm_class" in inspect.signature(hf_hub_download).parameters:
        kwargs["tqdm_class"] = _make_progress_tqdm(progress_callback)
        yield
        return

    hub_tqdm_mod = importlib.import_module("huggingface_hub.utils.tqdm")
    with _HUB_TQDM_PATCH_LOCK:
        original = hub_tqdm_mod.tqdm
        patched = _make_progress_tqdm(progress_callback, base=original)
        hub_tqdm_mod.tqdm = patched
        try:
            yield
        finally:
            hub_tqdm_mod.tqdm = original


class _Downloader(Protocol):
    def download(self, repo_id: str, filename: str,
                 revision: Optional[str] = None,
                 progress_callback: Optional[Callable[[int, int], None]] = None,
                 ) -> Path: ...


class _HFDownloader:
    """Thin wrapper around ``huggingface_hub.hf_hub_download``.

    Files land in HF's cache at ``HF_HOME`` (pinned by ``yamabiko.hf_env``
    to ``%LOCALAPPDATA%\\Yamabiko\\hf-cache\\``). Second calls for the
    same file return the cached path without a transfer.

    Retry on transient errors, ETag checks and xet chunk dedup are handled
    by huggingface_hub. We add byte-level progress, via
    ``_hub_progress_hook`` (``tqdm_class=`` on hub >= 1.1.0, the
    ``huggingface_hub.utils.tqdm`` rebind on hubs without the kwarg), and
    the pinned SHA-256 check of the landed file (``models.pinned_sha256``).

    ``cache_dir`` overrides HF's cache location. Production always uses
    the default (HF_HOME pinned by ``yamabiko.hf_env``); tests pass a
    fresh directory to force a real transfer (a cache hit produces no
    progress callbacks, which would let a cancel-path test pass without
    exercising cancellation).
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._cache_dir = cache_dir

    def download(self, repo_id: str, filename: str,
                 revision: Optional[str] = None,
                 progress_callback: Optional[Callable[[int, int], None]] = None,
                 ) -> Path:
        from huggingface_hub import hf_hub_download

        _log.info("hf_hub_download start: repo=%s file=%s revision=%s",
                  repo_id, filename, revision)
        t0 = time.monotonic()
        kwargs: dict = {
            "repo_id": repo_id,
            "filename": filename,
            "revision": revision,
        }
        if self._cache_dir is not None:
            kwargs["cache_dir"] = str(self._cache_dir)
        progress_cm = (
            _hub_progress_hook(progress_callback, kwargs)
            if progress_callback is not None
            else contextlib.nullcontext()
        )
        try:
            with progress_cm:
                path = Path(hf_hub_download(**kwargs))
        except Exception as exc:
            elapsed = time.monotonic() - t0
            _log.exception(
                "hf_hub_download FAILED after %.1fs: repo=%s file=%s revision=%s error=%r",
                elapsed, repo_id, filename, revision, exc,
            )
            raise
        elapsed = time.monotonic() - t0
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        _log.info(
            "hf_hub_download OK: repo=%s file=%s size=%d bytes elapsed=%.1fs landed=%s",
            repo_id, filename, size, elapsed, path,
        )
        expected = _pinned_sha256(repo_id, filename, revision)
        if expected is not None:
            actual = _file_sha256(path)
            if actual != expected:
                # Delete the blob itself (the snapshot entry may be a
                # symlink to it): a damaged file left in place would be
                # returned again by every later hf_hub_download call.
                target = Path(path).resolve()
                _log.error(
                    "checksum mismatch for %s/%s: expected %s got %s; deleting %s",
                    repo_id, filename, expected, actual, target,
                )
                try:
                    target.unlink()
                except OSError as exc:
                    _log.warning("could not delete damaged file %s: %s", target, exc)
                raise _ModelArtifactCorrupt(
                    f"checksum mismatch for {filename}", path=str(target),
                )
        return path


class _DownloadWorker(QObject):
    """Runs the download queue sequentially on a QThread.

    Emits per-file + overall signals so the main-thread page can update
    its widgets via Qt's queued connections (safe across threads).
    """

    file_started = Signal(int, str)        # (index, filename)
    file_finished = Signal(int, str)       # (index, filename)
    queue_finished = Signal(bool)          # success
    # (bytes_done, bytes_total) per file. 64-bit on purpose: the default-tier
    # GGUF is ~5.1 GiB, so both counts cross 2**31 mid-download and a plain
    # C++ int emit dies in libshiboken (Overflow warning + OverflowError).
    progress = Signal("qulonglong", "qulonglong")

    def __init__(self, specs: List[DownloadSpec],
                 downloader: Optional[_Downloader] = None) -> None:
        super().__init__()
        self._specs = specs
        self._downloader = downloader if downloader is not None else _HFDownloader()
        self.error: Optional[Exception] = None
        self.error_spec: Optional[DownloadSpec] = None
        # Best-effort cancel flag. Checked between files, and the
        # progress callback raises on it to abort mid-file where the
        # transfer path allows (HTTP yes, xet no — see
        # _DownloadCancelledError).
        self._cancelled: bool = False
        # The cancel raise fires at most ONCE per queue: since
        # huggingface_hub 1.19 the XetSession reports every callback
        # raise via sys.excepthook while continuing the transfer, so
        # per-tick raising prints a traceback per tick for the rest of
        # the in-flight file. One raise is all the HTTP path needs to
        # abort; the xet path stops at the between-files check anyway.
        self._cancel_raised: bool = False

    def cancel(self) -> None:
        """Request the queue stop ASAP. Thread-safe (single bool write)."""
        self._cancelled = True

    def run(self) -> None:
        _log.info("download queue starting: %d files", len(self._specs))
        for i, spec in enumerate(self._specs):
            if self._cancelled:
                _log.info("download queue cancelled by user at file %d/%d",
                          i, len(self._specs))
                self.queue_finished.emit(False)
                return
            _log.info("[%d/%d] %s %s", i + 1, len(self._specs),
                      spec.repo_id, spec.filename)
            self.file_started.emit(i, spec.filename)

            # Throttled progress logger — one sample per
            # _PROGRESS_LOG_INTERVAL_SECONDS so a stall is visible in
            # main-pipeline.log without spamming thousands of lines.
            last_log = [0.0]

            def _on_progress(done: int, total: int) -> None:
                if self._cancelled:
                    if not self._cancel_raised:
                        self._cancel_raised = True
                        # Aborts hf_hub_download mid-file on the HTTP
                        # path; the xet path reports this via
                        # sys.excepthook (hub >=1.19) / swallows it
                        # (older) and finishes the file.
                        raise _DownloadCancelledError()
                    # Post-cancel ticks (xet continuing the transfer):
                    # no re-raise (excepthook spam), no progress emit
                    # (the page would keep animating a cancelled
                    # download).
                    return
                now = time.monotonic()
                if now - last_log[0] >= _PROGRESS_LOG_INTERVAL_SECONDS:
                    last_log[0] = now
                    if total > 0:
                        pct = done * 100 / total
                        _log.info(
                            "  progress: %s %d/%d bytes (%.1f%%)",
                            spec.filename, done, total, pct,
                        )
                self.progress.emit(done, total)

            try:
                self._downloader.download(
                    spec.repo_id, spec.filename,
                    revision=spec.revision,
                    progress_callback=_on_progress,
                )
            except Exception as exc:
                if self._cancelled:
                    # The cancel raise (or any error racing a cancel) is
                    # a user decision, not a download failure: error
                    # stays None so the caller treats it as a cancel
                    # (Settings' dialog settles as cancelled; the wizard
                    # page ignores a queue it released), not a failure.
                    # Checked on the flag rather than
                    # the exception type because hf_hub_download may
                    # wrap what the progress callback raised.
                    _log.info(
                        "download queue cancelled mid-file at %s/%s",
                        spec.repo_id, spec.filename,
                    )
                    self.queue_finished.emit(False)
                    return
                # HF's internal retry has already exhausted. User-facing
                # Retry button offers a fresh queue run; cached files
                # from this partial run no-op on the second pass.
                _log.exception(
                    "download failed for %s/%s (rev=%s); aborting queue",
                    spec.repo_id, spec.filename, spec.revision,
                )
                self.error = exc
                self.error_spec = spec
                self.queue_finished.emit(False)
                return
            self.file_finished.emit(i, spec.filename)
        if self._cancelled:
            # Cancel raced the final file's completion — the xet path
            # always finishes the in-flight file, so a cancel during
            # the LAST file lands here instead of the between-files
            # check. Cancelled != success: the caller treats it as a
            # cancel, not a failure, and any files that DID land
            # no-op via the HF cache on the next run.
            _log.info(
                "download queue cancelled; in-flight file had already "
                "completed"
            )
            self.queue_finished.emit(False)
            return
        _log.info("download queue complete: all %d files OK", len(self._specs))
        self.queue_finished.emit(True)


def _progress_percent(done: int, total: int) -> int:
    """Clamp a byte-progress fraction to a sane ``[0, 100]`` integer percent.

    The upstream progress signal is best-effort, not exact. On the xet
    download path ``hf_xet`` reports ``group_report.total_bytes_completed``
    against a tqdm ``total`` pinned to the file's ``expected_size`` (see
    huggingface_hub ``file_download.xet_get``), and that pairing is known to
    drift: the running total can over-count (so ``done > total``) or jump,
    especially across resume/retry -- huggingface_hub#3234 ("total size
    incorrectly increasing after resuming"), #3266, xet-core#409. Rendered
    raw, a drifting total can read over 100% or, in principle, negative. The
    Rust xet core owns the accounting -- we can't fix it there, so we clamp
    at the consumer where the number is shown.
    """
    if total <= 0:
        return 0
    return max(0, min(100, int(done * 100 / total)))


# The step's rows: each queued file belongs to one (L5).
ROW_NAMES = ("Speech recognition", "Translation model", "Voice detection")
_SPEECH_ROW, _TRANSLATION_ROW, _VOICE_ROW = range(3)

WAITING = "Waiting"
DONE = "Done"
PAUSED = "Paused"
WAITING_FOR_OTHER = "Waiting for the other download to finish\u2026"
DOWNLOAD_STOPPED = "The download stopped."
DOWNLOAD_STOPPED_HELP = "Check your internet connection, then click Retry."
DELETE_FAILED = "Couldn't delete the old Yamabiko models."
DELETE_FAILED_HELP = "Free up space on the drive yourself, then click Retry."


def _row_of(spec: DownloadSpec) -> int:
    if spec.repo_id == _WHISPER_REPO:
        return _SPEECH_ROW
    if spec.repo_id == _SILERO_REPO:
        return _VOICE_ROW
    return _TRANSLATION_ROW


def _size_text(size: int) -> str:
    """A row's size in decimal units: "1.5 GB", "5.1 GB", "2 MB"."""
    if size >= 1_000_000_000:
        return _model_fetch().size_text(size / 1e9)
    return f"{max(1, round(size / 1e6))} MB"


class _QueueRelay(QObject):
    """One queue's worker signals on their way to the page, tagged with that
    queue's generation (Decision 130). It lives in the GUI thread (a child
    of the page), so the worker's signals reach it queued, as they reached
    the page before. Qt still delivers a signal the worker queued before a
    restart disconnected it; the page ignores it by its generation."""

    def __init__(self, page: "DownloadsPage", generation: int) -> None:
        super().__init__(page)
        self._page = page
        self._generation = generation

    def file_started(self, idx: int, filename: str) -> None:
        self._page._on_file_started(self._generation, idx, filename)

    def file_finished(self, idx: int, filename: str) -> None:
        self._page._on_file_finished(self._generation, idx, filename)

    def progress(self, done: int, total: int) -> None:
        self._page._on_progress(self._generation, done, total)

    def queue_finished(self, ok: bool) -> None:
        self._page._on_queue_finished(self._generation, ok)


class DownloadsPage(QWizardPage):
    """The Download step: every model the runtime needs into the HF cache
    under ``%LOCALAPPDATA%\\Yamabiko\\hf-cache\\``, with the overall amount
    and time left, one row per model, and the stopped states in an alert.

    Leaving the step never cancels (L3): only the wizard's Cancel / close
    (``shutdown_thread``) stops a download. ``railProgressChanged`` drives
    the rail's line under Download on every step.
    """

    # (fraction 0..1, stopped) while a queue runs or stopped unfinished;
    # (None, False) once it succeeded. The shell connects it to
    # StepRail.set_download.
    railProgressChanged = Signal(object, bool)

    def __init__(
        self,
        downloader: Optional[_Downloader] = None,
        *,
        palette: dict = MIST,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.setTitle("Downloading models")
        self.setSubTitle("This happens once. The models stay on your PC.")
        self._downloader = downloader
        self._time_left = _model_fetch()._TimeLeft(clock)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 26, 0, 0)
        layout.setSpacing(0)

        top = QHBoxLayout()
        top.setSpacing(12)
        amount_font = ui_font(14)
        amount_font.setWeight(QFont.Weight.DemiBold)
        self.amount = QLabel("")
        self.amount.setFont(amount_font)
        self.time_left = QLabel("")
        self.time_left.setObjectName("muted")
        self.time_left.setFont(ui_font(14))
        top.addWidget(self.amount)
        top.addStretch(1)
        top.addWidget(self.time_left)
        layout.addLayout(top)
        layout.addSpacing(8)
        self.overall = ProgressLine(palette, radius=0)
        layout.addWidget(self.overall)

        # The stopped states (disk full, any other failure) in one alert,
        # 20 px under the bar; hidden while the queue runs and after success.
        self.alert_box = QWidget()
        box = QVBoxLayout(self.alert_box)
        box.setContentsMargins(0, 20, 0, 0)
        self.alert = Alert(palette)
        box.addWidget(self.alert)
        self.delete_button = self.alert.add_button("Delete old Yamabiko models", primary=True)
        self.delete_button.clicked.connect(self._on_delete_clicked)
        self.retry_button = self.alert.add_button("Retry")
        self.retry_button.clicked.connect(self._on_retry_clicked)
        self.copy_button = self.alert.add_link("Copy details")
        self.copy_button.clicked.connect(self._on_copy_details_clicked)
        self.alert_box.hide()
        layout.addWidget(self.alert_box)

        layout.addSpacing(22)
        self.rows = StatusRows(ROW_NAMES, palette)
        for row in range(len(ROW_NAMES)):
            self.rows.set_row(row, "waiting", (Run(WAITING, ui_font(14)),))
        layout.addWidget(self.rows)
        layout.addStretch(1)

        self._thread: Optional[QThread] = None
        self._worker: Optional[_DownloadWorker] = None
        # The current queue's signal relay and generation (Decision 130):
        # whatever an earlier queue still has queued for this page is stale.
        self._relay: Optional[_QueueRelay] = None
        self._generation: int = 0
        self._success: bool = False
        self._completed: bool = False
        # Specs of the most recently started queue. Compared on re-entry so
        # a tier change on Quality rebuilds the queue (L3).
        self._last_specs: List[DownloadSpec] = []
        # Retired-model revisions and v0.1 model folders the last disk-full
        # state found: what "Delete old Yamabiko models" deletes.
        self._retired_revisions: tuple = ()
        self._legacy_dirs: tuple = ()
        # The redacted text Copy details puts on the clipboard.
        self._details: str = ""
        # Progress of the current queue: each spec's size (pinned; 0 until
        # known), the finished indexes, the running index and its clamped
        # bytes, and whether the queue stopped unfinished.
        self._sizes: List[int] = []
        self._finished: set = set()
        self._current: Optional[int] = None
        self._current_done: int = 0
        self._stopped: bool = False

    def _start_downloads(self) -> None:
        """Build the queue, reset the step, check the disk, start the thread."""
        # A previous queue may still be draining (a tier change while it
        # ran, or Retry while a cancelled thread is stuck mid-file). Detach
        # it first so it can't stay parented to the page or keep driving
        # the widgets; the release also makes whatever it still has queued
        # for this page stale (Decision 130).
        self._release_running_thread(wait_ms=0)

        specs = download_queue()
        self._last_specs = specs
        self._completed = False
        self._success = False
        self._sizes = [
            _pinned_size(s.repo_id, s.filename, s.revision) or 0 for s in specs
        ]
        self._finished = set()
        self._current = None
        self._current_done = 0
        self._stopped = False
        self._time_left.reset()
        self._retired_revisions = ()
        self._legacy_dirs = ()
        self._details = ""
        self.alert_box.hide()
        self.time_left.setText("")
        # Partial blobs from an earlier cancelled or killed run go first:
        # they are never resumed, and the space they hold counts against
        # the check below.
        _sweep_incomplete_blobs(spec.repo_id for spec in specs)
        # Disk check BEFORE spawning the worker: an ENOSPC otherwise
        # surfaces only 20-60 min into the multi-GB GGUF. On a shortfall the
        # queue does not start; Retry runs the check again.
        check = _space_check(specs)
        if check is not None and check.shortfall > 0:
            _log.warning(
                "download queue not started: %d bytes short on %s",
                check.shortfall, check.cache_root,
            )
            self._show_disk_full(check)
            return
        # If the detach above left a still-draining worker (blocked inside an
        # uninterruptible hf_hub_download), HF's per-blob lock can serialize
        # this new worker behind that drain. Say so; the first file start or
        # progress tick clears it.
        if thread_keeper.any_running():
            self.time_left.setText(WAITING_FOR_OTHER)
        self._refresh()
        self.completeChanged.emit()

        self._worker = _DownloadWorker(specs, downloader=self._downloader)
        self._relay = _QueueRelay(self, self._generation)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.file_started.connect(self._relay.file_started)
        self._worker.file_finished.connect(self._relay.file_finished)
        self._worker.progress.connect(self._relay.progress)
        self._worker.queue_finished.connect(self._relay.queue_finished)
        self._thread.start()

    def _release_running_thread(self, wait_ms: int) -> None:
        """Stop-or-detach the current worker thread, then null the refs.

        Cancels the worker, asks the thread's event loop to quit, and
        waits up to ``wait_ms``. A thread still running after that is
        blocked inside ``hf_hub_download`` (uninterruptible -- see
        _DownloadCancelledError) and gets detached into
        ``thread_keeper`` instead of staying parented to the page,
        because destroying a running QThread is a Qt 6 qFatal abort.
        The detached worker's signals are disconnected so the doomed queue
        can't keep driving the page widgets. Every release makes the
        released queue stale (Decision 130): a signal it queued before the
        release still arrives, with an old generation. A joined queue's
        relay and thread go with it (deleteLater; nothing can reach them any
        more).
        """
        self._generation += 1
        thread, worker, relay = self._thread, self._worker, self._relay
        self._thread = None
        self._worker = None
        self._relay = None
        if thread is None:
            return
        if worker is not None:
            worker.cancel()
        thread.quit()
        if thread.wait(wait_ms):
            if relay is not None:
                relay.deleteLater()
            thread.deleteLater()
            return
        if worker is not None and relay is not None:
            try:
                worker.file_started.disconnect(relay.file_started)
                worker.file_finished.disconnect(relay.file_finished)
                worker.progress.disconnect(relay.progress)
                worker.queue_finished.disconnect(relay.queue_finished)
            except RuntimeError:
                # Already disconnected (paranoia; connections are made
                # exactly once in _start_downloads).
                pass
        thread_keeper.release(thread, worker)

    def shutdown_thread(self, wait_ms: int = 2000) -> None:
        """Wizard-teardown hook: stop the queue or detach its thread.

        Called by YamabikoWizard before the page is destroyed (cancel,
        close, or accept). The only path that stops a download (L3). See
        _release_running_thread for the semantics.
        """
        self._release_running_thread(wait_ms=wait_ms)

    def initializePage(self) -> None:  # noqa: N802 -- Qt convention
        # Re-entry (L3): the same queue is left as it is -- running,
        # stopped (Retry restarts it) or done. A different queue (the tier
        # changed on Quality) restarts; _start_downloads detaches a worker
        # that still runs. Already-cached files no-op via HF's ETag check.
        if self._last_specs and download_queue() == self._last_specs:
            return
        self._start_downloads()

    def _on_file_started(self, generation: int, idx: int, filename: str) -> None:
        if generation != self._generation:
            return  # an earlier queue's (Decision 130)
        self._current = idx
        self._current_done = 0
        if self.time_left.text() == WAITING_FOR_OTHER:
            self.time_left.setText("")
        self._refresh()

    def _on_file_finished(self, generation: int, idx: int, filename: str) -> None:
        if generation != self._generation:
            return  # an earlier queue's (Decision 130)
        self._finished.add(idx)
        self._refresh()

    def _on_progress(self, generation: int, done: int, total: int) -> None:
        idx = self._current
        if generation != self._generation or idx is None or total <= 0:
            return
        if self._sizes[idx] <= 0:
            self._sizes[idx] = int(total)  # no size pin: the hub's own
        # The xet counter can over-count or jump (see _progress_percent):
        # clamp to the file's size so nothing reads past 100 %.
        self._current_done = max(0, min(int(done), self._sizes[idx]))
        left = self._time_left.update(self._done_bytes(), self._total_bytes())
        self.time_left.setText(
            _model_fetch().format_time_left(left) if left is not None else ""
        )
        self._refresh()

    def _total_bytes(self) -> int:
        return sum(self._sizes)

    def _done_bytes(self) -> int:
        done = sum(self._sizes[i] for i in self._finished)
        if self._current is not None and self._current not in self._finished:
            done += self._current_done
        return done

    def _refresh(self) -> None:
        """Paint the amount, the bar, the rows and the rail line from the
        byte counts."""
        total = self._total_bytes()
        done = self._done_bytes()
        fraction = done / total if total > 0 else 0.0
        self.amount.setText(_model_fetch().format_amount(done, total) if total > 0 else "")
        self.overall.set_fraction(fraction)
        self.overall.set_stopped(self._stopped)
        for row in range(len(ROW_NAMES)):
            self._refresh_row(row)
        if self._completed and self._success:
            self.railProgressChanged.emit(None, False)
        else:
            self.railProgressChanged.emit(fraction, self._stopped)

    def _refresh_row(self, row: int) -> None:
        indexes = [i for i, spec in enumerate(self._last_specs) if _row_of(spec) == row]
        size = sum(self._sizes[i] for i in indexes)
        done = sum(self._sizes[i] for i in indexes if i in self._finished)
        running = self._current in indexes and self._current not in self._finished
        if running:
            done += self._current_done
        size_label = _size_text(size) if size > 0 else ""
        font = ui_font(14)
        if indexes and all(i in self._finished for i in indexes):
            self.rows.set_row(row, "done", (Run(DONE, font),), size=size_label)
        elif running:
            percent = f"{_progress_percent(done, size)}%"
            bar = done / size if size > 0 else 0.0
            if self._stopped:
                self.rows.set_row(row, "paused", (Run(percent, font),), size=size_label,
                                  bar=bar, bar_stopped=True)
            else:
                self.rows.set_row(row, "now", (Run(percent, font, "ink"),), size=size_label,
                                  bar=bar)
        else:
            self.rows.set_row(row, "waiting", (Run(WAITING, font),), size=size_label)

    def _on_queue_finished(self, generation: int, ok: bool) -> None:
        worker = self._worker
        if generation != self._generation:
            # A released queue's result (a restart, the wizard closing, or
            # the cancel that release sent), queued before the release:
            # never this queue's (Decision 130). Leaving the step never
            # cancels (L3), so there is nothing to show.
            return
        self._success = ok
        self._completed = True
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        if ok:
            self.time_left.setText("")
            self._refresh()
        elif _is_disk_full(worker.error):
            # The partial file is never resumed: drop it now, as Retry would,
            # so the shortfall below counts the space it held (L6).
            _sweep_incomplete_blobs(spec.repo_id for spec in self._last_specs)
            self._show_disk_full(_space_check(self._last_specs), worker.error)
        else:
            self._show_failure(worker.error)
        self.completeChanged.emit()

    def _show_disk_full(
        self, check: Optional[_SpaceCheck], error: Optional[BaseException] = None,
    ) -> None:
        """The one disk-full state (L6) for a blocked check and a
        mid-download ENOSPC. Old Yamabiko models in Yamabiko's own cache
        and the v0.1 model folders count against the very space it asks
        for; when there are any, their deletion is the primary action."""
        if _hf_home_is_ours():
            self._retired_revisions, retired_bytes = _scan_retired_cache_entries()
        else:
            # A shared cache is never scanned: its retired repos are not
            # offered for deletion (the v0.1 folders below still are).
            self._retired_revisions, retired_bytes = (), None
        # The v0.1 folders count only on the drive the check measured:
        # deleting them anywhere else frees nothing it asks for.
        if check is None or _drive_letter(_legacy_models_root()) == check.drive:
            self._legacy_dirs, legacy_bytes = _scan_legacy_model_dirs()
        else:
            self._legacy_dirs, legacy_bytes = (), 0
        has_old = bool(self._retired_revisions or self._legacy_dirs)
        self.alert.set_text(_disk_full_message(
            check.drive if check is not None else "",
            check.shortfall if check is not None else 0,
        ))
        self.delete_button.setText(_old_models_label((retired_bytes or 0) + legacy_bytes))
        self.delete_button.setVisible(has_old)
        self.delete_button.setEnabled(True)
        set_primary(self.delete_button, has_old)
        set_primary(self.retry_button, not has_old)
        self.retry_button.setEnabled(True)
        self._details = _disk_full_details(check, retired_bytes, error, legacy_bytes=legacy_bytes)
        self._stop()

    def _show_failure(self, error: Exception) -> None:
        """Any failure but a full disk (L6): the network, the server, a
        damaged file."""
        self._retired_revisions = ()
        self._legacy_dirs = ()
        self.alert.set_text(DOWNLOAD_STOPPED, DOWNLOAD_STOPPED_HELP)
        self.delete_button.hide()
        set_primary(self.retry_button, True)
        self.retry_button.setEnabled(True)
        self._details = redact_user(
            f"{_friendly_error(error)}\n{self._format_error_details()}"
        )
        self._stop()

    def _stop(self) -> None:
        """The one stopped look (Decision 78): "Paused", the bar, the
        running row and the rail line muted, the alert shown."""
        self._stopped = True
        self.time_left.setText(PAUSED)
        self.alert_box.show()
        self._refresh()
        self.completeChanged.emit()

    def _on_retry_clicked(self) -> None:
        """Retry after a stop: the whole start path again (disk check
        included). Already-cached files come back from the cache (pinned
        ones after their digest check), so only the failed file and the
        ones after it transfer."""
        _log.info("user clicked Retry; restarting download queue")
        self._start_downloads()

    def _on_delete_clicked(self) -> None:
        """Delete old Yamabiko models: remove the retired-model revisions and
        the v0.1 model folders the disk-full state found, then run the whole
        start path again so the freed space is re-measured.

        Deletes EXACTLY what the alert was computed against -- never a
        fresh scan, which could pick up entries the user was never shown.
        Runs on the GUI thread (Decision 82: an unlink of a 4 GiB file took
        0.52 s on the dev box); the buttons are disabled and painted first.
        A failure (locked blob, permissions) keeps the user on the
        disk-full state with the failure named.
        """
        revisions = self._retired_revisions
        legacy = self._legacy_dirs
        if not revisions and not legacy:
            return
        self.delete_button.setEnabled(False)
        self.retry_button.setEnabled(False)
        QApplication.processEvents()  # paint the disabled buttons before the blocking delete
        try:
            if revisions:
                _delete_retired_cache_revisions(revisions)
            if legacy:
                _delete_legacy_model_dirs(legacy)
        except Exception as exc:
            _log.exception("freeing old Yamabiko models failed")
            self.alert.set_text(DELETE_FAILED, DELETE_FAILED_HELP)
            # After the disk facts the state was built on, not instead of them.
            self._details += "\n" + redact_user(f"{DELETE_FAILED}\n{type(exc).__name__}: {exc}")
            self.delete_button.setEnabled(True)
            self.retry_button.setEnabled(True)
            return
        self._start_downloads()

    def _on_copy_details_clicked(self) -> None:
        """Copy the stopped state's report (already redacted) with the
        flushing copy: Cancel can end the wizard in os._exit, which drops an
        unflushed clipboard (phase 6)."""
        from yamabiko.ui.error_window import copy_to_clipboard

        if self._details:
            copy_to_clipboard(self._details)

    def _format_error_details(self) -> str:
        """Build a paste-ready error string for bug reports."""
        from yamabiko import __version__

        worker = self._worker
        spec = worker.error_spec if worker else None
        raw_exc = worker.error if worker else None
        return (
            f"Yamabiko v{__version__} download error\n"
            f"File: {spec.filename if spec else 'unknown'}\n"
            f"Repo: {spec.repo_id if spec else 'unknown'}\n"
            f"Revision: {spec.revision if spec else 'unknown'}\n"
            f"Error: {raw_exc!r}\n"
            f"Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}"
        )

    def isComplete(self) -> bool:  # noqa: N802 -- Qt convention
        return self._completed and self._success
