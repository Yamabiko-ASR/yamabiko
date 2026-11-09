"""Shared atomic text-file write.

os.replace journals the NTFS rename atomically but does NOT flush data
blocks: without an fsync before the rename, power loss can publish a
zero-length/partial file under the final name. Same hardening as
config_writer._atomic_write and glossary._save_unlocked; this is the
generalized text form for new call sites.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically. On any failure the temp file
    is removed and the error re-raised; ``path`` is never left truncated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
