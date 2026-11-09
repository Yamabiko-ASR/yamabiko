"""Global hotkey registration for Yamabiko.

Wraps the `keyboard` library so tests can swap in a fake. Three supported
bindings: pause/resume, show/hide overlay, and show the series-search
palette (config keys in HOTKEY_KEYS). The `keyboard` hook never fails for a
combo another app holds, so `combo_taken` asks Windows directly (the
RegisterHotKey API other apps use for global shortcuts); HotkeyManager skips
a taken combo and reports it through `on_conflict`, like any other binding
failure. Settings uses the same probe for its "Used by another app" state.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from typing import Callable, Optional, Protocol

_log = logging.getLogger(__name__)

HOTKEY_KEYS = ("pause_resume", "toggle_overlay", "show_series")
MODIFIERS = ("Ctrl", "Alt", "Shift", "Win")
_MOD_ALIASES = {"ctrl": "Ctrl", "control": "Ctrl", "alt": "Alt", "shift": "Shift",
                "win": "Win", "windows": "Win"}
# RegisterHotKey modifier flags and the no-repeat flag.
_MOD_FLAGS = {"Alt": 0x0001, "Ctrl": 0x0002, "Shift": 0x0004, "Win": 0x0008}
_MOD_NOREPEAT = 0x4000
_ERROR_HOTKEY_ALREADY_REGISTERED = 1409
_PROBE_ID = 0xB0B0


class HotkeyTaken(Exception):
    """Another app already holds this combo as a system-wide shortcut."""


class _KeyboardBackend(Protocol):
    """Minimal surface — intentionally narrow so FakeBackend stays trivial
    and tests don't depend on OS-level keyboard hooks. Do NOT expand this
    to mirror the full `keyboard` module API."""
    def add_hotkey(self, combo: str, callback: Callable[[], None]) -> object: ...
    def clear_all_hotkeys(self) -> None: ...


def _default_backend() -> _KeyboardBackend:
    import keyboard
    return keyboard  # type: ignore[return-value]


def _normalize_combo(combo: str) -> str:
    return "".join(ch for ch in combo.lower() if not ch.isspace())


def _main_key(part: str) -> Optional[str]:
    up = part.upper()
    if len(up) == 1 and up.isascii() and (up.isalpha() or up.isdigit()):
        return up
    digits = up[1:]
    if up[:1] == "F" and digits.isascii() and digits.isdecimal() and 1 <= int(digits) <= 24:
        return f"F{int(digits)}"
    return None


def parse_combo(combo: str) -> Optional[tuple[frozenset[str], str]]:
    """Split "Ctrl+Alt+P" into ({"Ctrl", "Alt"}, "P").

    A bindable shortcut is one main key (A-Z, 0-9 or F1-F24) plus at least
    one of Ctrl, Alt, Win; an F-key may also stand alone or with Shift.
    Anything else (a bare or Shift-only letter, two main keys, other keys)
    is None."""
    if not isinstance(combo, str):
        return None
    mods: set[str] = set()
    key: Optional[str] = None
    for raw in combo.split("+"):
        part = raw.strip()
        if not part:
            return None
        mod = _MOD_ALIASES.get(part.lower())
        if mod is not None:
            if mod in mods:
                return None
            mods.add(mod)
            continue
        if key is not None:
            return None
        key = _main_key(part)
        if key is None:
            return None
    if key is None:
        return None
    if len(key) == 1 and not mods & {"Ctrl", "Alt", "Win"}:
        return None
    return frozenset(mods), key


def format_combo(mods, key: str) -> str:
    """"Ctrl+Alt+Shift+Win+K": modifiers in a fixed order, then the key."""
    return "+".join([m for m in MODIFIERS if m in mods] + [key])


def canonical_combo(combo: str) -> Optional[str]:
    """``combo`` in its stored spelling ("alt+ctrl+p" -> "Ctrl+Alt+P"), or
    None when it is not bindable."""
    parsed = parse_combo(combo)
    return format_combo(*parsed) if parsed is not None else None


def combo_label(combo: object) -> str:
    """A shortcut as the tray and setup show it: its stored spelling
    (canonical_combo), a hand-written combo that doesn't parse as written,
    "" for none (a missing, non-text or whitespace-only combo)."""
    if not isinstance(combo, str) or not combo.strip():
        return ""
    return canonical_combo(combo) or combo.strip()


def shortcut_labels(cfg: dict) -> dict[str, str] | None:
    """The global shortcuts the pipeline binds, from the merged config, as
    the tray and setup's Done step show them (phase-4 Decisions 33 and 60).
    None when hotkeys.enabled is false: the pipeline binds none. Otherwise
    each bound key mapped to its combo_label; empty combos are left out, and
    show_series only with the series feature on."""
    # Imported here: the gazetteer pulls in httpx, which hotkey binding
    # alone never needs.
    from yamabiko.gazetteer.gazetteer import feature_enabled

    hotkeys = cfg.get("hotkeys") or {}
    if not isinstance(hotkeys, dict) or not hotkeys.get("enabled", True):
        return None
    labels: dict[str, str] = {}
    for key in HOTKEY_KEYS:
        if key == "show_series" and not feature_enabled(cfg):
            continue
        label = combo_label(hotkeys.get(key))
        if label:
            labels[key] = label
    return labels


def _virtual_key(key: str) -> int:
    if len(key) == 1:
        return ord(key)  # VK_A..VK_Z and VK_0..VK_9 equal their ASCII codes
    return 0x70 + int(key[1:]) - 1  # VK_F1 = 0x70


def combo_taken(combo: str) -> bool:
    """True when another app holds ``combo`` as a system-wide shortcut.

    Registers the combo with Win32 RegisterHotKey and releases it at once;
    ERROR_HOTKEY_ALREADY_REGISTERED means taken. False for an unbindable
    combo, off Windows, or on any other failure."""
    parsed = parse_combo(combo)
    if parsed is None or sys.platform != "win32":
        return False
    mods, key = parsed
    flags = _MOD_NOREPEAT
    for mod in mods:
        flags |= _MOD_FLAGS[mod]
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    if user32.RegisterHotKey(None, _PROBE_ID, flags, _virtual_key(key)):
        user32.UnregisterHotKey(None, _PROBE_ID)
        return False
    return ctypes.get_last_error() == _ERROR_HOTKEY_ALREADY_REGISTERED


class HotkeyManager:
    """Manage the app's global hotkey bindings.

    Usage:
        mgr = HotkeyManager()
        mgr.register(
            pause="Ctrl+Alt+P", toggle_overlay="Ctrl+Alt+O",
            show_series="Ctrl+Alt+S",
            on_pause=app.toggle_pause, on_toggle_overlay=app.toggle_overlay,
            on_show_series=app.show_series_palette,
            on_conflict=app.notify_hotkey_conflict (taken combos arrive as HotkeyTaken)
        )
        ...
        mgr.unregister()
    """

    def __init__(self, backend: Optional[_KeyboardBackend] = None,
                 probe: Optional[Callable[[str], bool]] = None) -> None:
        self._backend = backend if backend is not None else _default_backend()
        self._probe = probe

    def register(
        self,
        *,
        pause: str,
        toggle_overlay: str,
        show_series: str,
        on_pause: Callable[[], None],
        on_toggle_overlay: Callable[[], None],
        on_show_series: Callable[[], None],
        on_conflict: Callable[[str, Exception], None] | None = None,
    ) -> None:
        bindings = [
            (pause, on_pause),
            (toggle_overlay, on_toggle_overlay),
            (show_series, on_show_series),
        ]
        for raw_combo, callback in bindings:
            if not raw_combo:
                continue
            try:
                combo = _normalize_combo(raw_combo)
                if (self._probe or combo_taken)(raw_combo):
                    raise HotkeyTaken(raw_combo)
                self._backend.add_hotkey(combo, callback)
                _log.info("hotkey registered: %s", combo)
            except HotkeyTaken as exc:
                _log.warning("hotkey %s is held by another app; not bound", raw_combo)
                if on_conflict is not None:
                    on_conflict(raw_combo, exc)
            except Exception as exc:
                _log.exception("hotkey %s failed", raw_combo)
                if on_conflict is not None:
                    on_conflict(raw_combo, exc)

    def unregister(self) -> None:
        try:
            self._backend.clear_all_hotkeys()
        except Exception as exc:
            _log.warning("clear_all_hotkeys failed: %s", exc)
