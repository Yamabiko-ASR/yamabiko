"""Atomic persistence boundary for a Settings Apply/OK operation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from yamabiko.config_writer import KeyPath


@dataclass(frozen=True)
class SettingsCommit:
    """Complete persisted and runtime state produced by one Settings save."""

    updates: Mapping[KeyPath, Any]
    remove_paths: tuple[KeyPath, ...]
    style: Mapping[str, Any]
    position: Optional[Mapping[str, Any]]


def persist_settings(commit: SettingsCommit) -> None:
    """Persist ``commit`` through one locked read/modify/replace cycle."""
    from yamabiko.config_writer import write_overrides

    write_overrides(
        commit.updates,
        remove_paths=commit.remove_paths,
    )
