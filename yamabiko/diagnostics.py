"""Field-diagnostics bundle collector.

One call, one artifact: zip every log under ``paths.logs_dir()``
(live ``main-*.log`` files, rotated ``.log.N`` backups, faulthandler
``crash-*.log`` evidence, the ``llama-server.log`` child capture,
``latency.log``) plus a redacted config/environment snapshot into
``Desktop/yamabiko-diagnostics-<ts>.zip`` -- something a beta user can
attach to a bug report in one drag.

Privacy contract (same line the startup banner draws): the snapshot
names the dotted override KEYS from ``user_overrides.yaml`` but never
their VALUES -- values can hold endpoints, paths, device names = PII.
Log files themselves honour the transcript-consent gating done at the
log-write sites, so shipping them verbatim is consistent with the
bug_report.yml privacy promise.

Resilience contract: :func:`collect_diagnostics` never raises into its
caller. It is wired to a tray click; a locked log file or an
unwritable Desktop must degrade to log evidence, not detonate the
overlay process.
"""
from __future__ import annotations

import logging
import platform
import sys
import time
import zipfile
from pathlib import Path

from yamabiko import __version__, gpu, paths

_log = logging.getLogger(__name__)

# Arcname of the redacted snapshot inside the zip. Tests and support
# docs reference it by name; keep in sync with tests/test_diagnostics.py.
_SNAPSHOT_NAME = "diagnostics.txt"


def _desktop_dir() -> Path:
    """Resolve the user's real Desktop folder.

    Uses ``SHGetKnownFolderPath(FOLDERID_Desktop)``: with OneDrive
    "Known Folder Move" active the Desktop lives under
    ``%USERPROFILE%\\OneDrive\\Desktop`` and the naive
    ``%USERPROFILE%\\Desktop`` guess points at a folder the user never
    looks at (or one that no longer exists). Verified live against
    shell32 on Windows 11 (2026-07-01). Falls back to the naive guess,
    then to the home dir, so the collector still lands the zip
    *somewhere* discoverable on any failure.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _GUID(ctypes.Structure):
                _fields_ = [
                    ("Data1", wintypes.DWORD),
                    ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8),
                ]

            # FOLDERID_Desktop {B4BFCC3A-DB2C-424C-B029-7FE99A87C641}
            folder_id = _GUID(
                0xB4BFCC3A, 0xDB2C, 0x424C,
                (ctypes.c_ubyte * 8)(
                    0xB0, 0x29, 0x7F, 0xE9, 0x9A, 0x87, 0xC6, 0x41,
                ),
            )
            buf = ctypes.c_wchar_p()
            hr = ctypes.windll.shell32.SHGetKnownFolderPath(
                ctypes.byref(folder_id), 0, None, ctypes.byref(buf),
            )
            if hr == 0 and buf.value:
                try:
                    return Path(buf.value)
                finally:
                    ctypes.windll.ole32.CoTaskMemFree(buf)
            # Non-exception failure (bad GUID/HRESULT, empty path): without
            # this line the API-level failure would be indistinguishable
            # from success-via-fallback in the logs.
            _log.warning(
                "SHGetKnownFolderPath(Desktop) returned hr=0x%08X "
                "(path %s); falling back to ~/Desktop",
                hr & 0xFFFFFFFF, "empty" if hr == 0 else "unset",
            )
        except Exception:
            _log.warning(
                "SHGetKnownFolderPath(Desktop) failed; falling back to "
                "~/Desktop", exc_info=True,
            )
    naive = Path.home() / "Desktop"
    return naive if naive.is_dir() else Path.home()


def _override_keys() -> list[str]:
    """Dotted key paths present in ``user_overrides.yaml`` -- KEYS only.

    Reads the sparse override file directly (not the merged config) so
    the snapshot shows exactly what the user changed. Any read/parse
    failure degrades to an empty list with a warning; the snapshot must
    never be the thing that breaks diagnostics collection.
    """
    try:
        import yaml

        p = paths.user_overrides_path()
        if not p.is_file():
            return []
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        _log.warning(
            "diagnostics: could not read override keys", exc_info=True,
        )
        return []

    out: list[str] = []

    def _walk(node: object, prefix: str = "") -> None:
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            dotted = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                _walk(v, dotted)
            else:
                # Leaf: record the key path, never the value.
                out.append(dotted)

    _walk(data)
    return sorted(out)


def _snapshot_text() -> str:
    """Banner-style environment info + override keys. Never raises."""
    lines = [
        f"yamabiko: {__version__}",
        f"mode: {'frozen' if getattr(sys, 'frozen', False) else 'dev'}",
        f"python: {platform.python_version()}",
        f"platform: {platform.platform()}",
    ]
    try:
        gpu_name = gpu.detect_gpu_name()
    except Exception:
        gpu_name = None
    lines.append(f"gpu: {gpu_name or 'not detected'}")
    keys = _override_keys()
    lines.append("override keys (values withheld):")
    if keys:
        lines.extend(f"  {k}" for k in keys)
    else:
        lines.append("  (none)")
    return "\n".join(lines) + "\n"


def collect_diagnostics(dest_dir: Path | None = None) -> Path:
    """Zip logs_dir()/*.log* (covers main-*.log + rotated backups,
    crash-*.log, llama-server.log, latency.log) + a redacted config
    snapshot (override KEYS + banner-style env info, no values) +
    version, to ``Desktop/yamabiko-diagnostics-<ts>.zip``.

    Returns the zip path. Never raises into the caller: a single
    unreadable log is skipped with a warning; a total failure is logged
    at ERROR, any partially-written zip is removed, and the returned
    path simply won't exist (callers ``exists()``-check it -- see the
    tray handler).
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(dest_dir) if dest_dir is not None else _desktop_dir()
    zip_path = out_dir / f"yamabiko-diagnostics-{ts}.zip"
    try:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        log_files = sorted(paths.logs_dir().glob("*.log*"))
        with zipfile.ZipFile(
            zip_path, "w", compression=zipfile.ZIP_DEFLATED,
        ) as zf:
            for f in log_files:
                try:
                    zf.write(f, arcname=f.name)
                except (OSError, ValueError):
                    # OSError: AV lock / mid-rotation rename. ValueError:
                    # zipfile refuses pre-1980 mtimes (strict_timestamps
                    # default). Either way, ONE bad file must not cost
                    # the user the rest of the bundle.
                    _log.warning(
                        "diagnostics: could not add %s; skipping",
                        f.name, exc_info=True,
                    )
            zf.writestr(_SNAPSHOT_NAME, _snapshot_text())
        _log.info("diagnostics bundle written: %s", zip_path)
    except Exception:
        # Unwritable Desktop, disk full, zipfile internals -- whatever
        # it was, the tray click must survive it. Evidence goes to the
        # log; the caller sees a path that doesn't exist.
        _log.exception("diagnostics collection failed (target: %s)", zip_path)
        # A failure AFTER ZipFile creation leaves a partial zip behind;
        # the caller's exists() check would announce that corrupt
        # half-bundle as success. Enforce the contract: on failure the
        # returned path must not exist. Best-effort -- the same disk
        # that broke the write may refuse the unlink too.
        try:
            zip_path.unlink(missing_ok=True)
        except OSError:
            _log.warning(
                "diagnostics: could not remove partial zip %s",
                zip_path, exc_info=True,
            )
    return zip_path
