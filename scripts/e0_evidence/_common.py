"""Shared validators and durable file primitives for the E0 evidence harness.

This module imports only the standard library, so every E0 module can depend
on it without an import cycle. Validators raise ``TypeError`` for a value of
the wrong type and ``ValueError`` for a malformed string. File primitives let
``OSError`` propagate so each caller can report its own artifact by name.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import uuid
from pathlib import Path

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_HASH_CHUNK_BYTES = 1024 * 1024


def require_uuid4(value: object, field: str) -> str:
    """Return ``value`` when it is a canonical (lowercase, hyphenated) UUIDv4."""

    message = f"{field} must be a canonical UUIDv4"
    if type(value) is not str:
        raise TypeError(message)
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(message) from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(message)
    return value


def require_sha256(value: object, field: str) -> str:
    """Return ``value`` when it is a lowercase hexadecimal SHA-256 digest."""

    message = f"{field} must be a lowercase SHA-256 hex digest"
    if type(value) is not str:
        raise TypeError(message)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(message)
    return value


def file_sha256_and_size(path: Path) -> tuple[str, int]:
    """Read ``path`` once and return its SHA-256 hex digest and byte count."""

    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of the bytes stored at ``path``."""

    return file_sha256_and_size(path)[0]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Durably replace ``path`` with ``data`` through a sibling temporary file.

    The parent directory must already exist. The temporary file never outlives
    the call, whether or not the replace succeeds.
    """

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            written = stream.write(data)
            if written != len(data):
                raise OSError("short atomic write")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def exclusive_write_bytes(path: Path, data: bytes) -> None:
    """Create ``path`` exclusively, owner-only, and make ``data`` durable in it.

    An existing file raises ``FileExistsError`` and is left untouched; any later
    failure removes the partially written file before the error propagates.
    """

    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    completed = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            written = stream.write(data)
            if written != len(data):
                raise OSError("short exclusive write")
            stream.flush()
            os.fsync(stream.fileno())
        completed = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not completed:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
