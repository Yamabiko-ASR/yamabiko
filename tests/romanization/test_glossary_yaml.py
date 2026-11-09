"""Tests for the shared YAML glossary loader."""
from __future__ import annotations

import logging
from pathlib import Path

from yamabiko.romanization.glossary_yaml import load_yaml_glossary


def test_load_valid_yaml(tmp_path: Path):
    p = tmp_path / "g.yaml"
    p.write_text(
        "version: 1\n"
        "entries:\n"
        "  田中:\n"
        "    en: Tanaka\n"
        "    reading: たなか\n",
        encoding="utf-8",
    )
    entries = load_yaml_glossary(p)
    assert "田中" in entries
    assert entries["田中"].en == "Tanaka"
    assert entries["田中"].reading == "たなか"


def test_load_missing_file_returns_empty(tmp_path: Path):
    assert load_yaml_glossary(tmp_path / "nope.yaml") == {}


def test_load_malformed_yaml_returns_empty(tmp_path: Path, caplog):
    p = tmp_path / "g.yaml"
    p.write_text(": not valid yaml :", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        result = load_yaml_glossary(p)
    assert result == {}
    assert any(
        "glossary" in r.getMessage().lower() or "yaml" in r.getMessage().lower()
        for r in caplog.records
    )


def test_load_missing_entries_key(tmp_path: Path):
    p = tmp_path / "g.yaml"
    p.write_text("version: 1\n", encoding="utf-8")
    assert load_yaml_glossary(p) == {}


# read_text ran inside the try block but only yaml.YAMLError was caught,
# so UnicodeDecodeError (non-UTF-8 hand-edited user.yaml) and OSError
# (sharing violation) escaped, propagated out of GlossaryManager.__init__,
# and main.py's catch-all disabled the ENTIRE romanization feature —
# contradicting the module docstring's "empty dict on any recoverable error".
# Seeded defect: narrow the except clause back to yaml.YAMLError only.

def test_load_non_utf8_file_returns_empty_and_warns(tmp_path: Path, caplog):
    p = tmp_path / "user.yaml"
    # Shift-JIS bytes for 田中 — invalid as UTF-8.
    p.write_bytes("entries:\n  田中:\n    en: Tanaka\n".encode("cp932"))
    with caplog.at_level(logging.WARNING):
        result = load_yaml_glossary(p)
    assert result == {}
    assert any("glossary" in r.getMessage().lower() for r in caplog.records)


def test_load_oserror_returns_empty_and_warns(tmp_path: Path, caplog):
    # A directory at the path makes read_text raise a real OSError
    # (PermissionError on Windows) without any monkeypatching.
    p = tmp_path / "user.yaml"
    p.mkdir()
    with caplog.at_level(logging.WARNING):
        result = load_yaml_glossary(p)
    assert result == {}
    assert any("glossary" in r.getMessage().lower() for r in caplog.records)


def test_load_missing_en_skipped(tmp_path: Path):
    p = tmp_path / "g.yaml"
    p.write_text(
        "version: 1\n"
        "entries:\n"
        "  broken:\n"
        "    reading: こわれた\n"
        "  田中:\n"
        "    en: Tanaka\n"
        "    reading: たなか\n",
        encoding="utf-8",
    )
    entries = load_yaml_glossary(p)
    assert "broken" not in entries
    assert "田中" in entries
