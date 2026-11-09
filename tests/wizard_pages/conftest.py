"""Shared fixtures for the wizard-page unit tests."""
from __future__ import annotations

import types

import pytest


@pytest.fixture(autouse=True)
def deterministic_space_preflight(monkeypatch):
    """Pin the DownloadsPage disk-space preflight's free-disk probe to a
    roomy deterministic value.

    The space check (``downloads._space_check``) reads the HOST's
    real free-disk bytes via ``shutil.disk_usage`` before every queue start;
    unstubbed, every existing DownloadsPage test would inherit a dependency
    on the developer/CI machine's actual disk state (a nearly-full runner
    would block queues that the tests expect to run). Rebinding the MODULE's
    ``shutil`` global (not the shared shutil module) scopes the pin to
    downloads.py. The preflight's own tests re-patch this same seam with
    their scenario values, so the real decision logic stays fully exercised
    there.
    """
    try:
        from yamabiko.wizard_pages import downloads as _downloads
    except Exception:  # PySide6-less environment: the Qt tests skip anyway
        yield
        return
    roomy = types.SimpleNamespace(total=2**40, used=0, free=2**40)
    monkeypatch.setattr(
        _downloads, "shutil",
        types.SimpleNamespace(disk_usage=lambda path: roomy),
    )
    yield
