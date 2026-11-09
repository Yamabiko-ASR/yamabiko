"""Problems the pipeline reports to its clients, and the copy they show.

Two kinds (spec Surface 5 > Notifications > Problems): the translator
stopped answering, and the audio device Yamabiko captured from went away.
The pipeline owns the active set (``AppState``) and sends it as a
``problems`` frame; the Qt overlay process and the browser/OBS page diff it
by kind. This module is the single source of the copy: plain, American
spelling, ASCII apostrophes, no em dashes.

A problem that ends in a pipeline restart (exit 44, 45 or 47) also leaves a
carry file. The watchdog takes it before the respawn, restarts the bundled
llama-server for a translation escalation, and hands the kinds to the next
pipeline in ``CARRY_ENV``, so that pipeline can tell new clients they
recovered: the Qt overlay process is the pipeline's child and starts over
with it. The carry also holds the shortcuts the pipeline announced as taken
(``CARRY_HOTKEYS_ENV`` for the next pipeline), so a restart does not
announce them again.

Only ``paths`` and ``fsutil`` are imported: the watchdog, the pipeline and
the overlay process all use this module.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from . import paths
from .fsutil import atomic_write_text

_log = logging.getLogger(__name__)

TRANSLATOR = "translator"
AUDIO = "audio"
# Display order everywhere: the frame, the notice lines, the tooltip.
KIND_ORDER: tuple[str, ...] = (TRANSLATOR, AUDIO)
RECOVERY: dict[str, str] = {
    TRANSLATOR: "Translations are back.",
    AUDIO: "Listening again.",
}
# The kinds the previous pipeline restarted on, comma separated (set by the
# watchdog for the next pipeline).
CARRY_ENV = "YAMABIKO_CARRIED_PROBLEMS"
# The shortcuts the previous pipeline announced as taken, as a JSON object of
# config key to combo (set by the watchdog for the next pipeline; plan
# Decision 72).
CARRY_HOTKEYS_ENV = "YAMABIKO_CARRIED_HOTKEYS"
_ENTRY_KEYS = ("kind", "title", "body", "line", "recovery")


def translator_problem(restarting: bool) -> dict:
    """The translator problem. ``restarting`` is True only when Yamabiko
    really restarts the translator (the bundled llama-server under the
    watchdog); otherwise the copy promises nothing (lead decision 4)."""
    if restarting:
        body = "The translator isn't responding. Yamabiko is restarting it."
        line = "The translator isn't responding. Restarting it\u2026"
    else:
        body = line = "The translator isn't responding."
    return {
        "kind": TRANSLATOR,
        "title": "Translations stopped",
        "body": body,
        "line": line,
        "recovery": RECOVERY[TRANSLATOR],
    }


def audio_problem(device: str) -> dict:
    """The audio problem for the endpoint whose friendly name is ``device``."""
    return {
        "kind": AUDIO,
        "title": "Can't hear your audio",
        "body": f"{device} was disconnected. Reconnect it, or pick another "
                "device in Settings.",
        "line": f"{device} was disconnected.",
        "recovery": RECOVERY[AUDIO],
    }


def ordered_kinds(kinds: Iterable) -> tuple[str, ...]:
    """The known kinds in ``kinds``, in display order, once each."""
    present = {kind for kind in kinds if isinstance(kind, str)}
    return tuple(kind for kind in KIND_ORDER if kind in present)


def is_problem(entry: object) -> bool:
    """True for a well-formed problem entry of a known kind."""
    return (
        isinstance(entry, dict)
        and entry.get("kind") in KIND_ORDER
        and all(isinstance(entry.get(key), str) for key in _ENTRY_KEYS)
    )


def make_problems_frame(active: Iterable[dict], recovered: Iterable[str] = ()) -> dict:
    """The ``problems`` frame: ``active`` entries in display order, and a
    recovery entry for each kind in ``recovered`` that is not active."""
    by_kind = {entry["kind"]: dict(entry) for entry in active}
    return {
        "type": "problems",
        "active": [by_kind[kind] for kind in KIND_ORDER if kind in by_kind],
        "recovered": [
            {"kind": kind, "recovery": RECOVERY[kind]}
            for kind in ordered_kinds(recovered) if kind not in by_kind
        ],
    }


@dataclass(frozen=True)
class Carry:
    """What a pipeline restarted on: the active problem kinds, whether the
    watchdog should restart the bundled llama-server, and the shortcuts it
    announced as taken (config key to combo)."""

    kinds: tuple[str, ...]
    restart_translator: bool
    hotkeys: dict[str, str] = field(default_factory=dict)


def carry_path() -> Path:
    return paths.app_data_dir() / "problem-carry.json"


def write_carry(kinds: Iterable[str], restart_translator: bool,
                hotkeys: Mapping[str, str] | None = None) -> None:
    """Leave the carry for the watchdog (atomic; raises on a write error)."""
    atomic_write_text(carry_path(), json.dumps({
        "kinds": list(ordered_kinds(kinds)),
        "restart_translator": bool(restart_translator),
        "hotkeys": dict(hotkeys or {}),
    }))


def take_carry() -> Carry | None:
    """Read and delete the carry; None when there is none or it is broken."""
    path = carry_path()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        _log.warning("problem carry unreadable; ignored", exc_info=True)
        return None
    try:
        path.unlink()
    except OSError:
        _log.warning("problem carry could not be removed", exc_info=True)
    try:
        # Bytes: a file that is not UTF-8 raises UnicodeDecodeError here, a
        # ValueError, so it is dropped like any other broken carry.
        data = json.loads(raw)
    except ValueError:
        _log.warning("problem carry is not JSON; ignored")
        return None
    if not isinstance(data, dict):
        return None
    kinds = data.get("kinds")
    return Carry(
        ordered_kinds(kinds if isinstance(kinds, list) else ()),
        data.get("restart_translator") is True,
        _clean_hotkeys(data.get("hotkeys")),
    )


def _clean_hotkeys(value: object) -> dict[str, str]:
    """Carried shortcuts from untrusted JSON: the string-to-string entries of
    an object; anything else reads as none."""
    if not isinstance(value, dict):
        return {}
    return {key: combo for key, combo in value.items()
            if isinstance(key, str) and isinstance(combo, str)}


def carry_env_value(kinds: Iterable[str]) -> str:
    return ",".join(ordered_kinds(kinds))


def kinds_from_env(value: str | None) -> tuple[str, ...]:
    return ordered_kinds((value or "").split(","))


def carry_hotkeys_env_value(hotkeys: Mapping[str, str]) -> str:
    return json.dumps(dict(hotkeys))


def hotkeys_from_env(value: str | None) -> dict[str, str]:
    try:
        data = json.loads(value or "{}")
    except ValueError:
        return {}
    return _clean_hotkeys(data)
