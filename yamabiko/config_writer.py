"""Sparse-write primitive for the user_overrides.yaml file.

Every persistent config change routes through :func:`write_overrides`.
The writer only stores values that *differ* from the shipped defaults;
setting a key back to its default prunes the key (and any now-empty
parents) from the overrides file, and if the final overrides tree is
empty the file is deleted entirely.

Writes are atomic and durable: tempfile + flush/fsync + ``os.replace``
Top-level key ordering is preserved via ``sort_keys=False``.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Collection, Mapping, Optional

import yaml
from filelock import FileLock, Timeout

_log = logging.getLogger(__name__)

# One click of Settings Apply/OK writes user_overrides.yaml from
# TWO processes at once — the overlay process persists audio/endpoint/
# quant keys while the pipeline process (WS set_style) persists style
# keys. Each write is a read-modify-write; without cross-process
# serialization an interleaving silently reverts a just-saved override
# (e.g. translation.quant_tier -> wrong GGUF quant at next launch).
# filelock is the established idiom (already shipped as a
# huggingface_hub dependency; HF uses it for exactly this on the cache).
_LOCK_TIMEOUT_S = 5.0

# Windows sharing violations: an antivirus scanner, the search indexer or a
# backup tool briefly opening user_overrides.yaml makes MoveFileEx / DeleteFile
# fail with ERROR_SHARING_VIOLATION or ERROR_ACCESS_DENIED (PermissionError).
# Those opens last milliseconds, so a short retry turns a dropped save into a
# saved one; a lasting lock still raises after the last retry.
_SHARING_RETRIES = 10
_SHARING_RETRY_SLEEP_S = 0.05


def _retry_on_sharing_violation(op, *args):
    """Run ``op(*args)``, retrying up to ``_SHARING_RETRIES`` times on
    PermissionError with a ``_SHARING_RETRY_SLEEP_S`` pause, then re-raise."""
    for attempt in range(_SHARING_RETRIES + 1):
        try:
            return op(*args)
        except PermissionError:
            if attempt == _SHARING_RETRIES:
                raise
            _log.debug("%s hit a sharing violation; retrying", getattr(op, "__name__", op))
            time.sleep(_SHARING_RETRY_SLEEP_S)


@contextmanager
def _locked_overrides(overrides_path: Path):
    """Hold the cross-process overrides writer lock for an RMW cycle.

    On ``Timeout`` (a wedged peer holding the lock), degrade to an
    UNLOCKED write with a warning: a Settings save must never become a
    silent no-op. The critical section is
    ~1 ms, so a 5 s timeout only fires when a peer process is stuck.
    """
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(overrides_path) + ".lock", timeout=_LOCK_TIMEOUT_S)
    # Acquire explicitly so the except covers ONLY acquisition: with the
    # `try: with lock: yield / except Timeout: yield` shape, a Timeout
    # raised from the BODY would reach this except and yield a second
    # time, which @contextmanager turns into an opaque
    # RuntimeError("generator didn't stop after throw()").
    acquired = False
    try:
        lock.acquire()
        acquired = True
    except Timeout:
        _log.warning(
            "could not acquire %s within %.1fs; writing unlocked so the "
            "save isn't dropped (peer process wedged?)",
            lock.lock_file, _LOCK_TIMEOUT_S,
        )
    try:
        yield
    finally:
        if acquired:
            lock.release()


# ---------------------------------------------------------------------------
# Atomic read/write helpers
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, data: dict) -> None:
    dir_ = path.parent
    dir_.mkdir(parents=True, exist_ok=True)
    # No mkstemp text=True and newline="\n" pinned: the O_TEXT fd plus a
    # text-mode fdopen (newline=None) each add a \r on Windows, writing
    # \r\r\n endings. fsutil.atomic_write_text pins the same way.
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(dir_)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            # Flush the user-space buffer, then fsync (Windows:
            # FlushFileBuffers) so the DATA is on disk before os.replace
            # publishes the name. os.replace/MoveFileEx journals the NTFS
            # rename atomically but does NOT flush data blocks — without
            # the fsync, power loss after the rename can leave a
            # zero-length/partial overrides file, which _read_yaml then
            # quarantines (silently resetting every user setting).
            f.flush()
            os.fsync(f.fileno())
        _retry_on_sharing_violation(os.replace, tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            # Swallow KEPT (the original write error below matters more
            # than the cleanup), but warn: silently accumulating .tmp
            # litter next to user_overrides.yaml was undiagnosable.
            _log.warning(
                "could not remove partial temp file %s after a failed "
                "write; leaving it behind", tmp_name, exc_info=True,
            )
        raise


def _quarantine_backup_path(path: Path) -> Path:
    """First free ``.corrupt(.N).bak`` sibling name for a quarantine.

    An existing backup is evidence of an EARLIER corruption and must not
    be clobbered by ``path.replace`` -- destroying it hid what a
    repeatedly-corrupted overrides file looked like the first time.
    """
    backup = path.with_suffix(path.suffix + ".corrupt.bak")
    n = 1
    while backup.exists():
        backup = path.with_suffix(path.suffix + f".corrupt.{n}.bak")
        n += 1
    return backup


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        # Reading inside the try: bytes that are not UTF-8 (UTF-16 from
        # Windows PowerShell 5.1 Out-File, CP932 from Notepad's ANSI
        # mode) raise UnicodeDecodeError, a ValueError, and must take the
        # same quarantine as unparseable YAML.
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        # The loader (config.load_full_config) already tolerates a corrupt
        # overrides file and degrades to defaults. The WRITER must not be
        # asymmetric: an unguarded safe_load here made every Settings/wizard
        # save raise -> swallowed by the UI's broad excepts -> silent no-op
        # Quarantine the bad file and start fresh so the write
        # actually persists.
        backup = _quarantine_backup_path(path)
        _log.warning(
            "corrupt %s (%s); backing up to %s and starting fresh so the "
            "write isn't silently dropped", path.name, exc, backup.name,
        )
        try:
            path.replace(backup)
        except OSError:
            # Swallow KEPT (the quarantine is best-effort; the atomic
            # write below overwrites the corrupt file in place anyway),
            # but warn: a silently-failing rename hid WHY no .corrupt.bak
            # ever appeared for a repeatedly-corrupted overrides file.
            _log.warning(
                "quarantine of corrupt %s to %s failed; the rewrite will "
                "overwrite it in place", path.name, backup.name,
                exc_info=True,
            )
        return {}
    if not isinstance(data, dict):
        # Valid YAML but not a mapping (hand-edit left a top-level list /
        # scalar). Same stakes as the parse-error branch above -- without
        # a quarantine, the next save rewrites the file with ONLY the new
        # key, silently resetting every other user setting and destroying
        # the evidence.
        backup = _quarantine_backup_path(path)
        _log.warning(
            "%s is valid YAML but not a mapping (%s); backing up to %s "
            "and starting fresh so the write isn't silently dropped",
            path.name, type(data).__name__, backup.name,
        )
        try:
            path.replace(backup)
        except OSError:
            _log.warning(
                "quarantine of non-mapping %s to %s failed; the rewrite "
                "will overwrite it in place", path.name, backup.name,
                exc_info=True,
            )
        return {}
    return data


# ---------------------------------------------------------------------------
# Path navigation
# ---------------------------------------------------------------------------

def _get_default(defaults: dict, key_path: list[str]) -> Any:
    """Return the shipped default at ``key_path`` or a sentinel if absent."""
    cur: Any = defaults
    for p in key_path:
        if not isinstance(cur, dict) or p not in cur:
            return _MISSING
        cur = cur[p]
    return cur


_MISSING = object()


def _set_in_tree(tree: dict, key_path: list[str], value: Any) -> None:
    cur = tree
    for p in key_path[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[key_path[-1]] = value


def _remove_path_and_prune(tree: dict, key_path: list[str]) -> None:
    """Remove ``key_path`` from ``tree`` and prune empty-dict ancestors."""
    # Walk down collecting the chain of containers for a post-order prune.
    stack: list[tuple[dict, str]] = []
    cur: Any = tree
    for p in key_path[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return
        stack.append((cur, p))
        cur = cur[p]
    if isinstance(cur, dict):
        cur.pop(key_path[-1], None)
    # Walk back up: if a container became empty, pop it from its parent.
    while stack:
        parent, k = stack.pop()
        child = parent.get(k)
        if isinstance(child, dict) and not child:
            parent.pop(k, None)
        else:
            break


def _apply_to_tree(
    overrides: dict, defaults: dict, key_path: list[str], value: Any,
    *, keep_default: bool = False,
) -> None:
    """Sparse-write semantics on the in-memory tree: set ``key_path``,
    or prune it (and empty ancestors) when ``value`` equals the
    shipped default and ``keep_default`` is False."""
    default_value = _get_default(defaults, key_path)
    if (
        not keep_default
        and default_value is not _MISSING
        and default_value == value
    ):
        _remove_path_and_prune(overrides, key_path)
    else:
        _set_in_tree(overrides, key_path, value)


def _flush_tree(overrides_path: Path, overrides: dict) -> None:
    """Persist the tree atomically; delete the file when it's empty."""
    if not overrides:
        try:
            _retry_on_sharing_violation(overrides_path.unlink)
        except FileNotFoundError:
            pass
        return
    _atomic_write(overrides_path, overrides)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

KeyPath = tuple[str, ...]


def write_overrides(
    updates: Mapping[KeyPath, Any],
    *,
    remove_paths: Collection[KeyPath] = (),
    keep_default_paths: Collection[KeyPath] = (),
) -> None:
    """Apply sparse updates and removals in one locked atomic write.

    Paths in ``keep_default_paths`` are written even when the value equals
    the shipped default: the file then records an explicit user choice
    instead of "never chosen".
    """
    normalized = [(tuple(path), value) for path, value in updates.items()]
    removals = [tuple(path) for path in remove_paths]
    keep = {tuple(path) for path in keep_default_paths}
    if any(not path for path, _value in normalized) or any(
        not path for path in removals
    ):
        raise ValueError("override paths must be non-empty")
    if not normalized and not removals:
        return

    from . import paths
    from . import config as _config  # lazy to avoid import cycle

    overrides_path = paths.user_overrides_path()
    defaults = _config._read_yaml(paths.shipped_defaults_path())
    with _locked_overrides(overrides_path):
        overrides = _read_yaml(overrides_path)
        for path, value in normalized:
            _apply_to_tree(
                overrides, defaults, list(path), value,
                keep_default=path in keep,
            )
        for path in removals:
            _remove_path_and_prune(overrides, list(path))
        _flush_tree(overrides_path, overrides)


def write_override(key_path: list[str] | KeyPath, value: Any) -> None:
    """Set ``key_path`` in user_overrides.yaml to ``value``.

    If the new value equals the shipped default at that path, the key is
    pruned from the overrides file instead (so the overrides file never
    accumulates redundant values and "reset to defaults" falls out for
    free). If pruning leaves an empty dict ancestor, the ancestor is
    pruned too. If the final overrides tree is empty, the overrides
    file is deleted entirely.

    The read-modify-write cycle holds the cross-process overrides lock
    so concurrent writers in the overlay and pipeline processes
    can't revert each other's just-saved keys.
    """
    write_overrides({tuple(key_path): value})


# ---------------------------------------------------------------------------
# Higher-level writers used by the wizard / settings dialog / browser
# ---------------------------------------------------------------------------


def write_overlay_style(style: dict) -> None:
    """Persist overlay.style as ONE locked read + ONE atomic write.

    Sparse semantics per key (defaults collapse individually, exactly
    as :func:`write_override` would do). Unknown style keys are
    silently skipped to avoid cluttering the overrides file with
    UI-only fields.

    Batching matters beyond efficiency: a per-key
    write_override loop runs a separate read-modify-write cycle per key
    from the pipeline process while the overlay process persists its own
    keys, and every cycle is a fresh race window. One locked RMW closes
    them all.
    """
    from . import paths
    from . import config as _config
    defaults = _config._read_yaml(paths.shipped_defaults_path())
    style_defaults = ((defaults.get("overlay") or {}).get("style") or {})
    items = {k: v for k, v in style.items() if k in style_defaults}
    write_overrides({
        ("overlay", "style", key): value
        for key, value in items.items()
    })


def write_overlay_position(position: dict) -> None:
    """Persist the overlay.position keys in ONE locked RMW and
    prune the legacy overlay.geometry override (its shipped default is
    null, so migration completion means the key must not linger)."""
    updates = {
        ("overlay", "position", key): position[key]
        for key in ("screen", "v_offset_pct", "h_offset_pct", "left_offset_pct")
        if key in position
    }
    write_overrides(
        updates,
        remove_paths=(("overlay", "geometry"),),
    )


def write_quant_tier(tier: str) -> None:
    """Persist the coupled quant-tier pair as ONE locked read + ONE atomic write.

    ``translation.quant_tier`` and ``asr.faster_whisper.compute_type`` are a
    single unit: the small tier exists to cut VRAM and the ASR compute type is
    switched with it (see :func:`yamabiko.gpu.tier_overrides`). Persisting them
    as two independent :func:`write_override` cycles lets a failure between the
    writes strand a mismatched pair on disk (a default tier alongside the small
    tier's compute type, or the reverse) with no retry. Batching both keys into
    one read-modify-write-flush means the pair lands together or not at all.

    Sparse semantics for the compute type (a value equal to the shipped
    default prunes the key, exactly as :func:`write_override` would). The
    tier itself is ALWAYS written, including ``default``: a pruned key is
    indistinguishable from "never chosen", and the wizard would then let
    GPU detection replace a deliberate Standard pick with ``small`` on an
    8 GB card. An unknown ``tier`` is coerced to ``default`` by
    :func:`~yamabiko.gpu.tier_overrides`.
    """
    from .gpu import tier_overrides  # lazy: avoids an import cycle

    patch = tier_overrides(tier)
    write_overrides(
        {
            ("translation", "quant_tier"): patch["translation"]["quant_tier"],
            ("asr", "faster_whisper", "compute_type"): (
                patch["asr"]["faster_whisper"]["compute_type"]
            ),
        },
        keep_default_paths=(("translation", "quant_tier"),),
    )


def write_audio_and_translation(
    audio_device_name: Optional[str],
    translation_endpoint: Optional[str],
) -> None:
    """Persist optional audio.device and translation.endpoint.

    The parameter is historically named ``audio_device_name`` because the
    value IS a device-name substring, but the config schema key is
    ``audio.device`` (matching ``default_config.yaml`` and
    ``main.py``'s ``cfg["audio"]["device"]`` read). Writing to
    ``audio.device_name`` — the prior bug — produced an orphan key that
    ``_walk_unknown_keys`` would strip on the next load with a warning.

    Each arg=None or empty string leaves its corresponding field
    untouched.
    """
    updates = {}
    if audio_device_name:
        updates[("audio", "device")] = audio_device_name
    if translation_endpoint:
        updates[("translation", "endpoint")] = translation_endpoint
    write_overrides(updates)
