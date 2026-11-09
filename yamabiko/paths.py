"""Centralised filesystem paths for Yamabiko.

The running app writes to `%LOCALAPPDATA%\\Yamabiko\\` — a user-writable
location that survives upgrades. Tests override via the
`YAMABIKO_APPDATA` environment variable.

Every filesystem-touching module goes through this file so rebrand,
relocation, and test isolation stay a one-module change.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_APP_NAME = "Yamabiko"


# Filename constants. Kept at module level so tests and adjacent modules
# don't drift out of sync on a rename.
_USER_OVERRIDES_NAME = "user_overrides.yaml"
_LEGACY_CONFIG_NAME = "config.yaml"
_LEGACY_BACKUP_NAME = "config.yaml.migrated.bak"


def app_data_dir() -> Path:
    """Root directory for app data. YAMABIKO_APPDATA overrides for tests."""
    override = os.environ.get("YAMABIKO_APPDATA")
    if override:
        return Path(override)
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        # Fallback for non-Windows test runners. Real shipping is Windows-only.
        local = str(Path.home() / ".local" / "share")
    return Path(local) / _APP_NAME


def bundle_data_dir() -> Path:
    """Root directory for read-only bundled data (glossaries, static/).

    In a PyInstaller-frozen bundle this is ``<bundle>/_internal/yamabiko``
    (the onedir's internal yamabiko package dir, where `File /r` from
    NSIS lands it via the Analysis spec). In dev mode it's the repo's
    yamabiko/ package dir resolved via ``__file__``.

    Separate from ``app_data_dir()``: app_data_dir is user-writable
    data that survives upgrades; bundle_data_dir is read-only ship
    content replaced on every install.

    Without this helper, config paths
    like ``romanization.glossaries_dir: yamabiko/glossaries`` resolved
    via ``Path(relative_string)`` which is cwd-dependent — works in
    dev mode (cwd = repo root) but silently broken in the frozen exe
    (cwd = install dir, but glossaries are under _internal/). Callers
    should use this helper to build absolute defaults.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass is not None:
            return Path(meipass) / "yamabiko"
        # Fallback: alongside yamabiko.exe
        return Path(sys.executable).resolve().parent / "_internal" / "yamabiko"
    # Dev mode: paths.py lives in the yamabiko package, so .parent IS
    # the package dir.
    return Path(__file__).resolve().parent


def config_path() -> Path:
    """Path to the legacy live config.yaml.

    DEPRECATED: kept only for the one-time migration branch in
    ``config.load_full_config`` and for backwards compatibility with a
    small number of call sites still being ported. New code should use
    :func:`user_overrides_path` for the sparse override file and
    :func:`shipped_defaults_path` for the authoritative schema.
    """
    return app_data_dir() / _LEGACY_CONFIG_NAME


def user_overrides_path() -> Path:
    """Path to the sparse user_overrides.yaml.

    The file is absent on a fresh install. It grows only when the user
    explicitly overrides a shipped default (via Settings, the wizard,
    or a hand edit). At load time :func:`config.load_full_config`
    deep-merges it onto the shipped defaults.
    """
    return app_data_dir() / _USER_OVERRIDES_NAME


def legacy_config_backup_path() -> Path:
    """Destination for the one-time migration's rename of the old config.yaml."""
    return app_data_dir() / _LEGACY_BACKUP_NAME


def shipped_defaults_path() -> Path:
    """Absolute path to the bundled ``default_config.yaml``.

    Resolves via :mod:`importlib.resources` so it works both in a source
    checkout and in a PyInstaller-frozen bundle. Never copied to
    user-writable space — read directly at load time.
    """
    try:
        from importlib.resources import files as _files
        ref = _files("yamabiko").joinpath("default_config.yaml")
        # as_file() is the robust path for zipped bundles, but on disk
        # the traversable is already a real filesystem path. Prefer str()
        # on the traversable since we only read it — avoids the context-
        # manager dance for the common case.
        p = Path(str(ref))
        if p.is_file():
            return p
    except Exception:
        # This helper can run BEFORE setup_main_log() wires handlers
        # (config load happens first). getLogger(__name__) is still the
        # right call: pre-setup the record reaches logging's lastResort
        # stderr handler in dev, and every post-setup occurrence lands in
        # the normal log. Swallow KEPT — a resources-probe failure must
        # fall back to the module-relative path, never crash config load.
        logging.getLogger(__name__).warning(
            "importlib.resources probe for default_config.yaml failed; "
            "falling back to the module-relative path",
            exc_info=True,
        )
    # Dev / PyInstaller --onedir fallback: the file lives next to this
    # module on disk.
    return Path(__file__).parent / "default_config.yaml"


def state_path() -> Path:
    """Path to the first-run wizard's state.json."""
    return app_data_dir() / "state.json"


def logs_dir() -> Path:
    """Directory where rotating logs (latency, main) are written."""
    return app_data_dir() / "logs"


def ensure_dirs() -> None:
    """Create app-data and logs directories if absent."""
    root = app_data_dir()
    for d in (root, root / "logs"):
        d.mkdir(parents=True, exist_ok=True)


def app_icon_path() -> Path:
    """Absolute path to the bundled app icon (multi-resolution .ico)."""
    return Path(__file__).resolve().parent.parent / "assets" / "icon.ico"
