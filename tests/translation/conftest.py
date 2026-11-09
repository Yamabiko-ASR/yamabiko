"""Shared fixtures for translation-package tests.

The polarity detector runs on the real fugashi / UniDic tagger (unidic-lite is
a bundled runtime dep, always present -- same as the romanization extractor
tests), so these are NOT `@pytest.mark.real`. One tagger is built per session
to avoid re-loading UniDic per test.
"""
from __future__ import annotations

import fugashi
import pytest


@pytest.fixture(scope="session")
def tagger() -> fugashi.Tagger:
    return fugashi.Tagger()
