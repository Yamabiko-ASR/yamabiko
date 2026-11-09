"""PipelineClient._on_text message-routing tests.

_on_text (ws_client.py) is pure JSON-routing logic — parse, dispatch on
"type", emit the matching Qt signal with extracted fields. It had ZERO
coverage: the frozen E2E asserts the SERVER-side broadcast dict, never
the Qt client's parse, so a one-character field-name typo (`"eng"` for
`"en"`) ships an overlay rendering empty English subtitles with every
suite green. These tests feed wire-format JSON strings through the real
_on_text and assert the exact signal payloads.

Signal recorders use direct Python connects: a direct `.connect`ed
slot runs synchronously inside the `.emit()` and its exceptions
propagate (PySide6 behaviour with direct connections), so no
event-loop pumping is needed.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("pytestqt")


@pytest.fixture
def client(qapp):
    from yamabiko.overlay_app.ws_client import PipelineClient

    c = PipelineClient(url="ws://127.0.0.1:1/ws")
    yield c
    c.stop()


@pytest.fixture
def recorders(client):
    """Connect a recorder to every routed signal; return the log dict."""
    rec: dict[str, list] = {
        "subtitle": [], "state": [], "style": [], "toggle": [],
    }
    client.subtitleReceived.connect(
        lambda jp, en, ruby: rec["subtitle"].append((jp, en, ruby))
    )
    client.stateReceived.connect(
        lambda genre, paused: rec["state"].append((genre, paused))
    )
    client.styleReceived.connect(lambda msg: rec["style"].append(msg))
    client.toggleOverlayReceived.connect(lambda: rec["toggle"].append(True))
    return rec


def _only(rec: dict, key: str) -> list:
    """Assert exactly one signal kind fired and return its payload list."""
    for k, v in rec.items():
        if k != key:
            assert v == [], f"unexpected {k} dispatch: {v}"
    return rec[key]


def test_subtitle_message_dispatches_jp_en_and_ruby_exactly(client, recorders):
    """A subtitle frame routes to subtitleReceived with the exact wire
    fields — jp, en, jp_ruby — and nothing else fires.

    Seeded-defect check: change `msg.get("en", "")` to `msg.get("eng", "")`
    in _on_text -> en arrives empty -> red. Or `t == "subtitle"` to
    `t == "subtitles"` -> nothing dispatches -> red.
    """
    ruby = [{"text": "田中", "reading": "たなか"}, {"text": "さん"}]
    client._on_text(json.dumps(
        {"type": "subtitle", "jp": "田中さん", "en": "Mr. Tanaka",
         "jp_ruby": ruby, "t": 12.5}
    ))
    assert _only(recorders, "subtitle") == [("田中さん", "Mr. Tanaka", ruby)]


def test_subtitle_message_defaults_missing_fields_to_empty(client, recorders):
    """Missing jp/en default to "" and a null jp_ruby reaches subscribers
    as [] — the overlay renders a blank row instead of crashing.

    Seeded-defect check: change `msg.get("jp", "")` to `msg["jp"]` ->
    KeyError propagates out of the direct emit -> red. (Note: dropping
    the `or []` alone is an EQUIVALENT mutation at this seam —
    Signal(str, str, list) marshals an emitted None to [] before the
    slot sees it — so the coercion contract is pinned at the subscriber
    side, not the expression.)
    """
    client._on_text(json.dumps({"type": "subtitle", "jp_ruby": None}))
    assert _only(recorders, "subtitle") == [("", "", [])]


def test_state_message_dispatches_genre_and_coerced_paused_bool(client, recorders):
    """A state frame routes to stateReceived with (genre, paused-as-bool);
    a missing genre defaults to "variety".

    Seeded-defect check: swap the emit args (paused, genre) -> red;
    change the genre default -> red.
    """
    client._on_text(json.dumps({"type": "state", "genre": "anime", "paused": 1}))
    client._on_text(json.dumps({"type": "state"}))
    assert _only(recorders, "state") == [("anime", True), ("variety", False)]
    assert recorders["state"][0][1] is True, "paused not coerced to bool"


def test_style_message_dispatches_full_payload_including_type(client, recorders):
    """A style frame routes the FULL message dict (the documented contract:
    styleReceived carries the payload including the "type" key — the Tray
    strips it).

    Seeded-defect check: emit a filtered dict (drop "type" / a field) -> red.
    """
    msg = {"type": "style", "jp_size": 22, "bg_color": "#123456"}
    client._on_text(json.dumps(msg))
    assert _only(recorders, "style") == [msg]


def test_toggle_overlay_message_dispatches_toggle_signal(client, recorders):
    """A toggle_overlay frame fires toggleOverlayReceived (no args).

    Seeded-defect check: delete the toggle_overlay branch -> red.
    """
    client._on_text(json.dumps({"type": "toggle_overlay"}))
    assert _only(recorders, "toggle") == [True]


def test_unknown_message_type_is_ignored_without_dispatch(client, recorders):
    """Unknown/missing "type" values fall through every branch: nothing
    fires, nothing raises (forward-compat: an older overlay must survive
    a newer pipeline's new frame kinds).

    Seeded-defect check: add a fall-through `else: self.styleReceived.emit(msg)`
    -> red.
    """
    client._on_text(json.dumps({"type": "totally_new_kind", "x": 1}))
    client._on_text(json.dumps({"no_type_at_all": True}))
    for k, v in recorders.items():
        assert v == [], f"unknown type dispatched to {k}: {v}"


def test_malformed_json_is_dropped_without_raising(client, recorders):
    """Garbage on the wire (truncated frame, non-JSON) is dropped silently
    — _on_text runs as a Qt slot off the socket; an uncaught
    JSONDecodeError would hit sys.excepthook on every bad frame.

    Seeded-defect check: remove the try/except json.JSONDecodeError ->
    this call raises -> red.
    """
    client._on_text("{not json")
    client._on_text("")
    for k, v in recorders.items():
        assert v == [], f"malformed frame dispatched to {k}: {v}"


# --- series routing: the state frame's new series fields + the palette cmd ---
#
# These connect their own recorders (not the shared `recorders` fixture)
# because a state frame now fires BOTH stateReceived and seriesReceived,
# which the fixture's `_only` single-dispatch invariant would reject.


def test_state_frame_also_dispatches_series_media_id_and_label(client):
    """The state frame carries series_media_id/series_label/series_native;
    _on_text emits seriesReceived(media_id, label, native) so the tray's
    'Series:' line and the overlay banner track the pipeline's authoritative
    series (on connect + after a change).

    Seeded-defect check: delete the seriesReceived.emit line in the state
    branch -> no series payload -> red; rename the series_media_id key ->
    the media_id arrives None -> red; drop series_native from the emit ->
    the Japanese title arrives "" -> red.
    """
    got: list = []
    client.seriesReceived.connect(
        lambda mid, label, native: got.append((mid, label, native))
    )
    client._on_text(json.dumps(
        {"type": "state", "genre": "anime", "paused": False,
         "series_media_id": 7, "series_label": "Frieren (2023)",
         "series_native": "葬送のフリーレン"}
    ))
    assert got == [(7, "Frieren (2023)", "葬送のフリーレン")]


def test_state_frame_without_series_keys_does_not_emit_series(client):
    """Consumer-side defense in depth: a state frame that OMITS the series
    keys (a legacy/third-party frame, or the pre-fix pause-hotkey frame that
    carried only genre+paused) must NOT be read as 'series cleared' --
    seriesReceived stays silent. An absent key is 'unspecified', not
    'cleared'. A frame that CARRIES the keys still emits. This is the guard
    that stops a keyless frame from resetting the tray's Series line to
    (none) while the pipeline gazetteer is active.

    Seeded-defect check: drop the `if "series_media_id" in msg` guard in
    _on_text's state branch -> the keyless frame emits seriesReceived(None,
    "", "") -> received != [] -> red.
    """
    received: list = []
    client.seriesReceived.connect(
        lambda mid, lbl, native: received.append((mid, lbl, native))
    )
    client._on_text(json.dumps(
        {"type": "state", "genre": "anime", "paused": False}
    ))
    assert received == [], (
        "a state frame without series keys must not emit seriesReceived"
    )
    client._on_text(json.dumps(
        {"type": "state", "genre": "anime", "paused": False,
         "series_media_id": 7, "series_label": "X"}
    ))
    assert received == [(7, "X", "")]


def test_state_frame_with_present_null_series_keys_emits_cleared_series(client):
    """The guard keys on PRESENCE, not truthiness: a state frame that
    carries series_media_id explicitly set to null (a fresh AppState's
    on-connect snapshot, or a genuine set_series(None, None) deselect) IS an
    explicit clear and must emit seriesReceived(None, "", "") so the tray
    shows (none). The label and native nulls are coerced to "" for the
    Signal(object, str, str) str slots.

    Seeded-defect check: change the guard from `if "series_media_id" in msg`
    to a value test like `if msg.get("series_media_id") is not None` -> a
    present-null clear no longer emits -> received stays [] -> red.
    """
    received: list = []
    client.seriesReceived.connect(
        lambda mid, lbl, native: received.append((mid, lbl, native))
    )
    client._on_text(json.dumps(
        {"type": "state", "genre": "anime", "paused": False,
         "series_media_id": None, "series_label": None, "series_native": None}
    ))
    assert received == [(None, "", "")]


def test_show_series_palette_frame_fires_palette_signal(client):
    """A show_series_palette frame (raised by the pipeline's Ctrl+Alt+S
    hotkey) fires showSeriesPaletteReceived with no args -> the overlay opens
    the palette.

    Seeded-defect check: delete the show_series_palette branch in _on_text ->
    nothing fires -> red.
    """
    got: list = []
    client.showSeriesPaletteReceived.connect(lambda: got.append(True))
    client._on_text(json.dumps({"type": "show_series_palette"}))
    assert got == [True]


def test_notice_frame_carries_its_settings_page(client):
    """A notice frame routes its title, body and Settings page; a frame
    without a page carries "".

    Seeded-defect check: drop settings_page in _on_text -> the tray cannot
    route the click -> red."""
    seen: list = []
    client.noticeReceived.connect(lambda title, body, page: seen.append((title, body, page)))
    client._on_text(json.dumps({"type": "notice", "title": "T", "body": "B",
                                "settings_page": "shortcuts"}))
    client._on_text(json.dumps({"type": "notice", "title": "T", "body": "B"}))
    assert seen == [("T", "B", "shortcuts"), ("T", "B", "")]


def test_problems_frame_routes_valid_entries_only(client):
    """A problems frame routes its active entries and recovered entries;
    malformed ones are dropped at the trust boundary and a frame without the
    lists routes two empty lists.

    Seeded-defect check: drop the problems branch in _on_text, or pass the
    entries through unchecked -> red."""
    from yamabiko import problems

    seen: list = []
    client.problemsReceived.connect(lambda active, recovered: seen.append((active, recovered)))
    good = problems.translator_problem(True)
    client._on_text(json.dumps({
        "type": "problems",
        "active": [good, {"kind": "audio", "title": 5}, "junk"],
        "recovered": [{"kind": "audio", "recovery": "Listening again."},
                      {"kind": "elsewhere", "recovery": "x"}],
    }))
    client._on_text(json.dumps({"type": "problems"}))
    assert seen == [
        ([good], [{"kind": "audio", "recovery": "Listening again."}]),
        ([], []),
    ]
