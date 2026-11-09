"""The problem contract (yamabiko/problems.py): copy, frame, restart carry."""
from __future__ import annotations

from yamabiko import problems
from yamabiko.state import AppState


def test_problem_copy_matches_the_spec():
    # Spec Surface 5 > Problems, exact; the copy without a restart promise is
    # lead decision 4. Seeded defects: a curly apostrophe, "..." for the
    # ellipsis, or the restart promise when nothing restarts -> red.
    assert problems.translator_problem(restarting=True) == {
        "kind": "translator",
        "title": "Translations stopped",
        "body": "The translator isn't responding. Yamabiko is restarting it.",
        "line": "The translator isn't responding. Restarting it\u2026",
        "recovery": "Translations are back.",
    }
    quiet = problems.translator_problem(restarting=False)
    assert quiet["body"] == quiet["line"] == "The translator isn't responding."
    assert problems.audio_problem("Headphones (USB Audio)") == {
        "kind": "audio",
        "title": "Can't hear your audio",
        "body": "Headphones (USB Audio) was disconnected. Reconnect it, or pick "
                "another device in Settings.",
        "line": "Headphones (USB Audio) was disconnected.",
        "recovery": "Listening again.",
    }


def test_problems_frame_orders_kinds_and_lists_carried_recoveries_only_in_the_window():
    # Translator above audio whatever the insertion order; a carried recovery
    # rides only a connect frame (now given) inside its window, and never for
    # a kind that is active again. Seeded defects: insertion order, recovered
    # on every frame, or no deadline -> red.
    state = AppState()
    assert state.set_problem(problems.audio_problem("Speakers")) is True
    assert state.set_problem(problems.translator_problem(True)) is True
    assert state.set_problem(problems.translator_problem(True)) is False
    state.set_carried_recoveries(["audio", "translator"], until=100.0)
    frame = state.problems_frame(now=99.0)
    assert [p["kind"] for p in frame["active"]] == ["translator", "audio"]
    assert frame["recovered"] == []
    assert state.clear_problem("translator") is True
    assert state.clear_problem("translator") is False
    assert state.problems_frame(now=99.0)["recovered"] == [
        {"kind": "translator", "recovery": "Translations are back."}]
    assert state.problems_frame(now=100.0)["recovered"] == []
    assert state.problems_frame()["recovered"] == []
    assert state.problem_kinds() == ("audio",)


def test_carry_is_read_once_and_a_broken_file_is_dropped(tmp_path, monkeypatch):
    # Decision 18: the watchdog takes the carry once; garbage never survives.
    # Seeded defects: take without deleting, or raise on bad JSON or on a
    # file that is not UTF-8 -> red.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    problems.write_carry(["audio", "translator"], restart_translator=True)
    assert problems.take_carry() == problems.Carry(("translator", "audio"), True)
    assert problems.take_carry() is None
    problems.carry_path().write_text("{not json", encoding="utf-8")
    assert problems.take_carry() is None
    assert not problems.carry_path().exists()
    problems.carry_path().write_bytes(b"\xff")
    assert problems.take_carry() is None
    assert not problems.carry_path().exists()
    assert problems.kinds_from_env("audio,bogus,translator") == ("translator", "audio")
    assert problems.carry_env_value(("audio", "translator")) == "translator,audio"


def test_carry_keeps_the_announced_shortcuts_and_a_broken_field_reads_as_none(tmp_path, monkeypatch):
    # Decision 72: the shortcuts a pipeline announced as taken ride the carry
    # and the watchdog's variable to the next pipeline; a broken field reads
    # as no shortcuts and keeps the rest of the carry. Seeded defects: drop
    # the field on write or on read, or raise on a broken one -> red.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    problems.write_carry([], False, {"pause_resume": "Ctrl+Alt+P"})
    assert problems.take_carry() == problems.Carry((), False, {"pause_resume": "Ctrl+Alt+P"})
    problems.carry_path().write_text('{"kinds": ["audio"], "hotkeys": ["Ctrl+Alt+P"]}',
                                     encoding="utf-8")
    assert problems.take_carry() == problems.Carry(("audio",), False, {})
    value = problems.carry_hotkeys_env_value({"pause_resume": "Ctrl+Alt+P"})
    assert problems.hotkeys_from_env(value) == {"pause_resume": "Ctrl+Alt+P"}
    assert problems.hotkeys_from_env(None) == {}
    assert problems.hotkeys_from_env("{not json") == {}
