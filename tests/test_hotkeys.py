"""Hotkeys module wraps `keyboard` for testability.

The real library registers OS-level hooks; tests pass a fake backend
via dependency injection so they run headless and fast.
"""
from __future__ import annotations

import pytest

from yamabiko import hotkeys


class FakeBackend:
    def __init__(self):
        self.registered: list[tuple[str, object]] = []
        self.cleared = False

    def add_hotkey(self, combo, callback):
        self.registered.append((combo, callback))

    def clear_all_hotkeys(self):
        self.cleared = True


def _never_taken(combo):
    return False


def test_register_binds_all_enabled_hotkeys():
    backend = FakeBackend()
    pause_cb = lambda: None
    toggle_cb = lambda: None
    series_cb = lambda: None
    manager = hotkeys.HotkeyManager(backend=backend, probe=_never_taken)
    manager.register(pause="Ctrl+Alt+P", toggle_overlay="Ctrl+Alt+O",
                     show_series="Ctrl+Alt+S",
                     on_pause=pause_cb, on_toggle_overlay=toggle_cb,
                     on_show_series=series_cb)
    combos = [c for c, _ in backend.registered]
    assert "ctrl+alt+p" in combos
    assert "ctrl+alt+o" in combos
    assert "ctrl+alt+s" in combos


def test_register_binds_show_series_and_invokes_its_callback():
    """The registered show_series callback, when the backend fires it,
    actually invokes on_show_series (not merely that it was passed) --
    §3.9 wired-not-exercised guard.

    Seeded defect: drop `(show_series, on_show_series)` from the bindings
    list in HotkeyManager.register -> "ctrl+alt+s" is never registered and
    the invocation assert goes red.
    """
    backend = FakeBackend()
    fired: list[str] = []
    manager = hotkeys.HotkeyManager(backend=backend, probe=_never_taken)
    manager.register(pause="Ctrl+Alt+P", toggle_overlay="Ctrl+Alt+O",
                     show_series="Ctrl+Alt+S",
                     on_pause=lambda: None, on_toggle_overlay=lambda: None,
                     on_show_series=lambda: fired.append("show"))
    by_combo = {c: cb for c, cb in backend.registered}
    assert "ctrl+alt+s" in by_combo, "show_series combo was not registered"
    by_combo["ctrl+alt+s"]()  # simulate the OS firing the hotkey
    assert fired == ["show"], (
        "the registered show_series callback did not invoke on_show_series"
    )


def test_register_skips_empty_combos():
    backend = FakeBackend()
    manager = hotkeys.HotkeyManager(backend=backend, probe=_never_taken)
    manager.register(pause="", toggle_overlay="Ctrl+Alt+O", show_series="",
                     on_pause=lambda: None, on_toggle_overlay=lambda: None,
                     on_show_series=lambda: None)
    combos = [c for c, _ in backend.registered]
    assert combos == ["ctrl+alt+o"]


def test_unregister_clears_all():
    backend = FakeBackend()
    manager = hotkeys.HotkeyManager(backend=backend, probe=_never_taken)
    manager.register(pause="Ctrl+Alt+P", toggle_overlay="", show_series="",
                     on_pause=lambda: None, on_toggle_overlay=lambda: None,
                     on_show_series=lambda: None)
    manager.unregister()
    assert backend.cleared is True


def test_register_surfaces_conflicts(monkeypatch):
    class ConflictBackend:
        def add_hotkey(self, combo, callback):
            raise ValueError(f"{combo} already registered by another app")
        def clear_all_hotkeys(self):
            pass
    manager = hotkeys.HotkeyManager(backend=ConflictBackend(), probe=_never_taken)
    errors: list[tuple[str, Exception]] = []
    manager.register(pause="Ctrl+Alt+P", toggle_overlay="", show_series="",
                     on_pause=lambda: None, on_toggle_overlay=lambda: None,
                     on_show_series=lambda: None,
                     on_conflict=lambda combo, exc: errors.append((combo, exc)))
    assert len(errors) == 1
    assert errors[0][0] == "Ctrl+Alt+P"  # original casing preserved
    assert isinstance(errors[0][1], ValueError)
    assert "already registered" in str(errors[0][1])


def test_normalize_combo_handles_case_and_spaces():
    assert hotkeys._normalize_combo("Ctrl+Alt+P") == "ctrl+alt+p"
    assert hotkeys._normalize_combo("ctrl + alt + p") == "ctrl+alt+p"
    assert hotkeys._normalize_combo("  Ctrl+Alt+O  ") == "ctrl+alt+o"


def test_parse_combo_takes_one_main_key_with_a_modifier_or_an_f_key():
    # Seeded defects: accept a bare letter (it would hijack typing), reject
    # digits or F-keys, keep the user's modifier order, let a non-ASCII
    # digit reach int() (ValueError), or keep an F-key's leading zero -> red.
    assert hotkeys.parse_combo("Ctrl+Alt+P") == (frozenset({"Ctrl", "Alt"}), "P")
    assert hotkeys.canonical_combo("alt + ctrl + 5") == "Ctrl+Alt+5"
    assert hotkeys.canonical_combo("shift+f24") == "Shift+F24"
    assert hotkeys.canonical_combo("Ctrl+F05") == "Ctrl+F5"
    assert hotkeys.canonical_combo("win+shift+k") == "Shift+Win+K"
    assert hotkeys.parse_combo("P") is None
    assert hotkeys.parse_combo("Shift+P") is None
    assert hotkeys.parse_combo("Ctrl+Alt") is None
    assert hotkeys.parse_combo("Ctrl+Alt+Space") is None
    assert hotkeys.parse_combo("Ctrl+P+Q") is None
    assert hotkeys.parse_combo("Ctrl+F²") is None


def test_register_skips_a_combo_another_app_holds_and_reports_it():
    # The keyboard hook never fails for a taken combo, so the probe decides.
    # Seeded defects: bind it anyway, or report it as a generic failure ->
    # red.
    backend = FakeBackend()
    errors = []
    manager = hotkeys.HotkeyManager(backend=backend, probe=lambda combo: combo == "Ctrl+Alt+S")
    manager.register(pause="Ctrl+Alt+P", toggle_overlay="", show_series="Ctrl+Alt+S",
                     on_pause=lambda: None, on_toggle_overlay=lambda: None,
                     on_show_series=lambda: None,
                     on_conflict=lambda combo, exc: errors.append((combo, exc)))
    assert [combo for combo, _ in backend.registered] == ["ctrl+alt+p"]
    assert len(errors) == 1 and errors[0][0] == "Ctrl+Alt+S"
    assert isinstance(errors[0][1], hotkeys.HotkeyTaken)
