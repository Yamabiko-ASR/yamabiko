"""ProblemTracker: what each problems frame changed (plan Decisions 18, 22, 43)."""
from __future__ import annotations

import pytest

from yamabiko import problems
from yamabiko.overlay_app.problem_view import ProblemTracker


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _record(tracker: ProblemTracker) -> list:
    events: list = []
    tracker.changed.connect(lambda active: events.append(("changed", [p["kind"] for p in active])))
    tracker.recovered.connect(lambda kind, text: events.append(("recovered", kind, text)))
    tracker.started.connect(lambda entry: events.append(("started", entry["kind"])))
    return events


def test_one_frame_emits_changed_then_recovered_then_started(qapp):
    """Decision 22: in one frame the translator recovers while the audio
    problem starts. The overlay drops a problem line before it adds the
    recovery line, and the toast comes last.

    Seeded defects: emit started before changed, or no recovery for a kind
    that left the set -> red."""
    tracker = ProblemTracker()
    tracker.apply([problems.translator_problem(True)], [])
    events = _record(tracker)
    tracker.apply([problems.audio_problem("Speakers")], [])
    assert events == [
        ("changed", ["audio"]),
        ("recovered", "translator", "Translations are back."),
        ("started", "audio"),
    ]
    assert tracker.active() == [problems.audio_problem("Speakers")]


def test_repeated_frames_stay_quiet_and_a_carried_recovery_shows_once(qapp):
    """The set survives reconnects: the same frame again emits nothing. A
    new device name for an active kind is a change, not a new problem (no
    toast). A carried recovery shows once per kind, never for an active
    kind (Decisions 18 and 43).

    Seeded defects: treat every frame as new, start on a content change,
    or forget which carried kinds were shown -> red."""
    tracker = ProblemTracker()
    events = _record(tracker)
    speakers = [problems.audio_problem("Speakers")]
    headphones = [problems.audio_problem("Headphones")]
    carried = [{"kind": "translator", "recovery": "Translations are back."},
               {"kind": "audio", "recovery": "Listening again."}]
    tracker.apply(speakers, [])
    tracker.apply(speakers, [])
    tracker.apply(headphones, [])
    tracker.apply(headphones, carried)
    tracker.apply(headphones, carried)
    assert events == [
        ("changed", ["audio"]),
        ("started", "audio"),
        ("changed", ["audio"]),
        ("recovered", "translator", "Translations are back."),
    ]
