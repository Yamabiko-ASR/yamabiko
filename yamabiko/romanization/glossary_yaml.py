"""Shared loader for YAML glossary files (user.yaml and _seed.yaml).

Returns a dict of surface -> GlossaryEntry, or empty dict on any
recoverable error. Logs WARNING for parse errors. Skips entries
missing required fields (en).
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .glossary import GlossaryEntry

_log = logging.getLogger(__name__)


def load_yaml_glossary(path: Path) -> dict[str, GlossaryEntry]:
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as e:
        # UnicodeDecodeError is a ValueError, not an OSError — listed
        # explicitly. read_text runs inside this try, so a mis-encoded or
        # unreadable user.yaml degrades to "layer empty" instead of
        # propagating out of GlossaryManager.__init__ and disabling the
        # whole romanization feature.
        _log.warning("glossary yaml load error at %s: %s", path, e)
        return {}
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("entries") or {}
    if not isinstance(entries, dict):
        return {}
    out: dict[str, GlossaryEntry] = {}
    for surface, data in entries.items():
        if not isinstance(data, dict):
            continue
        en = data.get("en")
        if not isinstance(en, str) or not en:
            continue
        reading = data.get("reading")
        out[surface] = GlossaryEntry(
            en=en,
            reading=reading if isinstance(reading, str) else "",
        )
    return out
