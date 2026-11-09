"""Tray menu-handler invocation tests.

Previously the entire Tray class was covered by a single
`isinstance(overlay.tray, Tray)` assertion -- every command-bearing handler
body was untested, so a regression (wrong action, missing command, the
graceful-quit wiring) would ship silently. These tests construct a Tray with
a fake PipelineClient that records send_command, invoke each handler, and
assert the observable effect.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("pytestqt")

from tests.overlay_app._qt_drain import drain_qt_debris


@pytest.fixture(autouse=True)
def _drain_qt_objects():
    """Destroy each test's Qt debris (notably the real download QThread spun by
    the quant-download quit-release case) within its own teardown -- see
    tests/overlay_app/_qt_drain.py for the native-teardown-flake rationale."""
    yield
    drain_qt_debris()


class _FakeClient:
    """Stands in for PipelineClient: records send_command; provides the
    Qt signals Tray connects to (as mocks -- the tests call the handlers
    directly rather than emitting). `connected` mirrors the real
    client's contract: send_command returns False (a dropped frame)
    while disconnected, True once connected."""

    def __init__(self):
        self.sent: list[dict] = []
        self.connected = True
        self.stateReceived = MagicMock()
        self.styleReceived = MagicMock()
        self.connectedChanged = MagicMock()
        self.seriesReceived = MagicMock()
        self.noticeReceived = MagicMock()

    def send_command(self, cmd: dict) -> bool:
        self.sent.append(cmd)
        return self.connected


def _tray(qapp, on_quit=None, overlay_window=None, palette=None,
          on_change_series=None, style_seed=None, auto_genre=True, **extra):
    from yamabiko.overlay_app.tray import Tray
    client = _FakeClient()
    tray = Tray(
        client, on_quit=on_quit or (lambda: None),
        overlay_window=overlay_window, palette=palette,
        on_change_series=on_change_series, style_seed=style_seed,
        auto_genre=auto_genre, **extra,
    )
    return tray, client


class _WindowSentinel:
    """A truthy stand-in for the single open SettingsWindow ref.

    The preview path only runs while a window is open (the belt in
    _on_style_preview no-ops when _settings_window is None). Tests that drive
    the preview handlers directly -- rather than through a real constructed
    window -- set tray._settings_window to one of these so they exercise the
    real (window-open) path. _on_style_preview only reads the ref for None-ness;
    _on_preview_ended additionally parks the closed window's last ref in a
    deleteLater timer (C12), so the sentinel carries a no-op deleteLater."""

    def deleteLater(self) -> None:  # parked by _on_preview_ended (C12)
        pass


def _make_signal_client_class():
    """Build a real-Signal stand-in for PipelineClient.

    Unlike _FakeClient's MagicMock signals — where .connect() is a no-op
    and .emit() reaches nobody — emissions through these typed Signals
    only arrive at Tray's handlers if Tray.__init__ actually performed
    the .connect() calls. Signatures mirror ws_client.PipelineClient
    (stateReceived(str, bool), styleReceived(dict),
    connectedChanged(bool)). Built lazily inside a factory so importing
    this module never touches PySide6 before the importorskip above.
    """
    from PySide6.QtCore import QObject, Signal

    class _FakeSignalClient(QObject):
        stateReceived = Signal(str, bool)
        styleReceived = Signal(dict)
        connectedChanged = Signal(bool)
        seriesReceived = Signal(object, str, str)
        noticeReceived = Signal(str, str, str)

        def __init__(self):
            super().__init__()
            self.sent: list[dict] = []
            self.connected = True

        def send_command(self, cmd: dict) -> bool:
            self.sent.append(cmd)
            return self.connected

    return _FakeSignalClient


def _tray_with_real_signals(qapp, overlay_window=None, on_change_series=None):
    from yamabiko.overlay_app.tray import Tray
    client = _make_signal_client_class()()
    tray = Tray(
        client, on_quit=lambda: None, overlay_window=overlay_window,
        on_change_series=on_change_series,
    )
    return tray, client


def _icon_image(icon, px: int):
    """The QIcon's pixmap at px x px device pixels as a comparable QImage."""
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QImage

    return icon.pixmap(QSize(px, px), 1.0).toImage().convertToFormat(
        QImage.Format.Format_ARGB32_Premultiplied)


# --- Signal WIRING — emissions (not direct handler calls) must
# --- reach the handlers. Every other test in this file invokes the
# --- handlers directly, which is wired-not-exercised for the
# --- three client.<signal>.connect(...) lines in Tray.__init__.


def test_state_signal_emission_drives_the_pause_label_and_the_flyout_genre(qapp):
    """Emitting stateReceived through a real Signal updates the menu —
    proving Tray.__init__ connected client.stateReceived to _on_state.

    Seeded defect: delete the `client.stateReceived.connect(self._on_state)`
    line in tray.py -> the emission reaches no slot, the pause label
    stays 'Pause' and the flyout's Genre segment does not flip -> red.
    """
    tray, client = _tray_with_real_signals(qapp)
    client.stateReceived.emit("anime", True)
    assert tray._pause_action.text() == "Resume", (
        "stateReceived emission did not reach _on_state; the "
        "stateReceived.connect() wiring is missing"
    )
    assert tray._flyout._genre.value() == "anime"


def test_style_signal_emission_restyles_overlay_window(qapp):
    """Emitting styleReceived through a real Signal restyles the live
    overlay window and updates _current_style — proving the
    styleReceived.connect(self._on_style_received) wiring exists.

    Seeded defect: delete the styleReceived.connect line in tray.py ->
    the window keeps the default style and _current_style is stale -> red.
    """
    from yamabiko.overlay_app.overlay_window import OverlayWindow
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    window = OverlayWindow()
    try:
        tray, client = _tray_with_real_signals(qapp, overlay_window=window)
        # Sentinel 40 differs from the default (14) AND stays inside the
        # jp_size range [8, 48], so the now-active ingestion clamp
        # (_on_style_received -> _clamp_style) passes it through unchanged and
        # the wiring claim stays observable. (An out-of-range marker would be
        # clamped and could no longer distinguish a wired handler from a stale
        # default.)
        client.styleReceived.emit(
            {"type": "style", **STYLE_DEFAULTS, "jp_size": 40}
        )
        assert window._style["jp_size"] == 40, (
            "styleReceived emission did not reach _on_style_received; the "
            "styleReceived.connect() wiring is missing"
        )
        assert tray._current_style["jp_size"] == 40
    finally:
        window.close()


def test_on_style_received_clamps_garbage_ws_frame_before_apply(qapp):
    """The WS style path is untrusted: a loopback set_style echo can carry
    padding:None, a non-numeric size, or a bool max_width_pct (json.loads even
    accepts NaN/Infinity). _on_style_received must run config._clamp_style on the
    frame BEFORE storing it as _current_style and applying it, so the garbage is
    coerced to shipped defaults and a subsequent add_subtitle can't detonate on
    setContentsMargins(None) / int(None) downstream. This is the single
    ingestion clamp for the overlay process (disk loads already clamp in
    load_overlay_config; WS frames did not until now).

    Seeded defect (one line): drop the ``_clamp_style(raw)`` call in
    _on_style_received (store/apply ``raw`` directly) -> padding stays None
    and the clamped-value assertions read the garbage instead of the
    defaults -> red.
    """
    from yamabiko.overlay_app.overlay_window import OverlayWindow
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    window = OverlayWindow()
    try:
        tray, _client = _tray(qapp, overlay_window=window)
        tray._on_style_received({
            "type": "style",
            **STYLE_DEFAULTS,
            "padding": None,
            "jp_size": "bad",
            "max_width_pct": True,
        })
        # Stored + applied style is the CLAMPED dict, not the raw frame.
        assert tray._current_style["padding"] == 10
        assert tray._current_style["jp_size"] == 16
        assert tray._current_style["max_width_pct"] == 70
        assert window._style["padding"] == 10
        # The whole point: rendering after a garbage frame must not raise.
        window.add_subtitle("字幕", "subtitle")
        assert window.stack.frame().lines, "add_subtitle produced no line"
    finally:
        window.close()


def test_connected_signal_emission_flushes_pending_offline_style(qapp, monkeypatch):
    """Emitting connectedChanged(True) through a real Signal triggers the
    pending-style flush — proving the
    connectedChanged.connect(self._on_connected_changed) wiring exists.

    Seeded defect: delete the connectedChanged.connect line in tray.py ->
    the reconnect emission reaches no slot, nothing is flushed -> red.
    """
    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    monkeypatch.setattr(
        tray_mod, "write_overlay_style", lambda style: None, raising=False,
    )
    tray, client = _tray_with_real_signals(qapp)
    client.connected = False
    tray._send_style_change({**STYLE_DEFAULTS, "jp_size": 99})

    client.connected = True
    n_before = len(client.sent)
    client.connectedChanged.emit(True)
    flushed = [
        c for c in client.sent[n_before:] if c.get("action") == "apply_style"
    ]
    assert flushed and flushed[-1]["style"]["jp_size"] == 99, (
        "connectedChanged emission did not reach _on_connected_changed; "
        "the connectedChanged.connect() wiring is missing"
    )


def test_pause_toggle_sends_pause_then_resume(qapp):
    """_on_pause_toggled sends 'pause' when running, and 'resume' once state
    reports paused -- the branch on self._paused must actually drive the
    command.

    Seeded-defect drill: hardcode the action to 'pause' -> the second call
    still sends 'pause' -> this fails.
    """
    tray, client = _tray(qapp)
    tray._on_pause_toggled()
    assert client.sent[-1] == {"type": "command", "action": "pause"}

    tray._on_state("variety", True)  # pipeline now reports paused
    tray._on_pause_toggled()
    assert client.sent[-1] == {"type": "command", "action": "resume"}


def test_genre_chosen_sends_set_genre(qapp):
    """_on_genre_chosen forwards the chosen genre as a set_genre command.

    Seeded-defect drill: drop the genre kwarg / send a constant -> fails.
    """
    tray, client = _tray(qapp)
    tray._on_genre_chosen("news")
    assert client.sent[-1] == {
        "type": "command", "action": "set_genre", "genre": "news",
    }


def test_quit_requested_sends_ws_quit_before_app_quit(qapp, qtbot):
    """Tray->Quit must send {action:quit} to the pipeline over the WS
    FIRST (so it drains + takes llama-server down), then defer the
    local QApplication quit via QTimer (not synchronous) -- AND the
    deferred quit must actually FIRE once the event loop runs (a bare
    `quit_calls == []` assertion is equally true when the QTimer line
    is deleted and the overlay process lives forever after Quit).

    Seeded-defect drill: remove the send_command(quit) line -> the WS quit is
    never sent -> this fails; call self._on_quit_cb() synchronously instead
    of via QTimer -> quit_calls is non-empty -> the deferral assertion fails;
    delete the QTimer.singleShot(500, ...) line outright -> the waitUntil
    times out -> this fails.
    """
    quit_calls: list[bool] = []
    tray, client = _tray(qapp, on_quit=lambda: quit_calls.append(True))
    tray._on_quit_cb = lambda: quit_calls.append(True)

    tray._on_quit_requested()

    assert {"type": "command", "action": "quit"} in client.sent, (
        "Tray->Quit did not send the WS quit command"
    )
    assert quit_calls == [], (
        "on_quit fired synchronously; it must be deferred via QTimer so the "
        "pipeline gets its drain window"
    )
    # Drive the event loop past the 500 ms deferral: the quit callback
    # must actually fire, exactly once.
    qtbot.waitUntil(lambda: quit_calls == [True], timeout=2000)


# --- Settings Apply while the WS is disconnected ---


def test_style_apply_while_disconnected_applies_without_second_persist(qapp, monkeypatch):
    """Clicking Apply while the WS is down (pipeline starting, or inside
    the up-to-30s reconnect backoff) used to be a silent no-op:
    send_command dropped the frame and the style apply + persistence
    both live server-side. The offline path must (1) restyle the live
    overlay window and update _current_style so re-opening Settings shows the
    change. Settings already persisted the transaction, so the tray must not
    write it again.

    Seeded defect: revert _send_style_change to the bare
    send_command(...) call -> the window keeps the old style, nothing
    is persisted, _current_style is stale -> all three assertions red.
    """
    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.overlay_app.overlay_window import OverlayWindow
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    persisted: list[dict] = []
    monkeypatch.setattr(
        tray_mod, "write_overlay_style", persisted.append, raising=False,
    )
    window = OverlayWindow()
    try:
        tray, client = _tray(qapp, overlay_window=window)
        client.connected = False

        tray._send_style_change({**STYLE_DEFAULTS, "jp_size": 99})

        assert window._style["jp_size"] == 99, (
            "offline Apply did not restyle the live overlay window"
        )
        assert persisted == [], (
            "offline Apply wrote the already-committed style a second time"
        )
        assert tray._current_style["jp_size"] == 99, (
            "offline Apply did not update _current_style; re-opening "
            "Settings would show the stale style"
        )
    finally:
        window.close()


def test_style_apply_while_disconnected_flushes_to_pipeline_on_reconnect(qapp, monkeypatch):
    """The offline style must also reach the pipeline once the WS comes
    back, or the server's on-connect style snapshot reverts the visual
    change (same-process reconnect case). The flush fires once; a second
    connect must not re-send.

    Seeded defect: drop the connectedChanged flush handler -> nothing
    new is sent on reconnect -> red.
    """
    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    monkeypatch.setattr(
        tray_mod, "write_overlay_style", lambda style: None, raising=False,
    )
    tray, client = _tray(qapp)
    client.connected = False
    tray._send_style_change({**STYLE_DEFAULTS, "jp_size": 99})
    n_before = len(client.sent)

    client.connected = True
    tray._on_connected_changed(True)

    flushed = client.sent[n_before:]
    assert flushed and flushed[-1]["action"] == "apply_style" and (
        flushed[-1]["style"]["jp_size"] == 99
    ), f"offline style was not flushed on reconnect; got {flushed}"

    # Flush is one-shot: a later reconnect must not replay it.
    n_after = len(client.sent)
    tray._on_connected_changed(True)
    assert len(client.sent) == n_after, "flush replayed on second connect"


def test_online_apply_supersedes_pending_offline_style(qapp, monkeypatch):
    """A successful online send clears any style still queued from an
    earlier offline apply. Otherwise a reconnect flush that raced the
    socket closing (pending kept) followed by a newer ONLINE apply
    would replay the stale offline style over the newer one on the next
    reconnect.

    Seeded defect: drop the `self._pending_style = None` from
    _send_style_change's sent branch -> the stale jp_size=99 replays
    after reconnect -> red.
    """
    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    monkeypatch.setattr(
        tray_mod, "write_overlay_style", lambda style: None, raising=False,
    )
    tray, client = _tray(qapp)

    # Offline apply queues a pending style...
    client.connected = False
    tray._send_style_change({**STYLE_DEFAULTS, "jp_size": 99})
    # ...the reconnect flush races the socket closing and fails, so
    # the pending style survives...
    tray._on_connected_changed(True)  # send_command still returns False
    # ...then a NEWER apply succeeds online.
    client.connected = True
    tray._send_style_change({**STYLE_DEFAULTS, "jp_size": 55})

    n_before = len(client.sent)
    tray._on_connected_changed(True)
    assert len(client.sent) == n_before, (
        "stale offline style replayed after a newer online apply; "
        "the successful send must clear _pending_style"
    )


def test_style_apply_while_connected_skips_second_persist(qapp, monkeypatch):
    """Settings owns persistence; the successful round-trip only applies and
    broadcasts, and nothing is queued for reconnect flush.

    Seeded defect: make _send_style_change persist unconditionally ->
    the recorder fires -> red.
    """
    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    persisted: list[dict] = []
    monkeypatch.setattr(
        tray_mod, "write_overlay_style", persisted.append, raising=False,
    )
    tray, client = _tray(qapp)
    client.connected = True

    tray._send_style_change({**STYLE_DEFAULTS, "jp_size": 55})
    assert persisted == [], (
        "online Apply wrote the already-committed style a second time"
    )

    n_before = len(client.sent)
    tray._on_connected_changed(True)
    assert len(client.sent) == n_before, (
        "online Apply left a pending style behind; reconnect replayed it"
    )


# --- On-connect snapshot vs reconnect flush order ---


def test_reconnect_flush_wins_over_stale_on_connect_style_snapshot(qapp, monkeypatch):
    """On reconnect the wire order at the Tray is:
    (1) connectedChanged(True) -> flush the pending offline style,
    (2) the server's on-connect STALE style snapshot lands ~one RTT later
        (transient flash -- documented as acceptable, cosmetic),
    (3) the server merges the flush and broadcasts -- the flush MUST win:
        the final applied/current style is the offline-applied one.

    The test plays the server faithfully: the merged broadcast in (3) is
    echoed from the flush frame actually sent in (1), so if the flush never
    fires the test goes red at step (1) instead of hand-feeding a frame the
    server would never have produced.

    Seeded defect: invert the guard in _on_connected_changed (always
    return) -> no flush is sent on reconnect -> red at the step-(1) assert.
    """
    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.overlay_app.overlay_window import OverlayWindow
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    monkeypatch.setattr(
        tray_mod, "write_overlay_style", lambda style: None, raising=False,
    )
    window = OverlayWindow()
    try:
        tray, client = _tray(qapp, overlay_window=window)
        client.connected = False
        # Sentinel 40 stays inside the jp_size range [8, 48] so the flush,
        # re-received through the now-active ingestion clamp in
        # _on_style_received, survives unchanged (an out-of-range marker would
        # be clamped and the "flush wins" claim could no longer be observed).
        tray._send_style_change({**STYLE_DEFAULTS, "jp_size": 40})

        # (1) reconnect: connectedChanged fires BEFORE any server frame
        # arrives (QWebSocket signals connect on handshake completion).
        client.connected = True
        n_before = len(client.sent)
        tray._on_connected_changed(True)
        flushes = [
            c for c in client.sent[n_before:] if c.get("action") == "apply_style"
        ]
        assert flushes, (
            "no style flush sent on reconnect; the server's stale on-connect "
            "snapshot would win and revert the user's offline Apply"
        )

        # (2) the stale on-connect snapshot lands. The transient flash is
        # the documented cosmetic trade-off; assert it so the contract is
        # explicit rather than accidental.
        tray._on_style_received({"type": "style", **STYLE_DEFAULTS})
        assert window._style["jp_size"] == STYLE_DEFAULTS["jp_size"]

        # (3) the server merges the flush and broadcasts it back: final
        # state must be the offline-applied style. Flush wins.
        tray._on_style_received({"type": "style", **flushes[-1]["style"]})
        assert window._style["jp_size"] == 40, (
            "reconnect flush did not win over the stale on-connect snapshot"
        )
        assert tray._current_style["jp_size"] == 40
    finally:
        window.close()


def test_stale_style_snapshot_does_not_cancel_pending_flush(qapp, monkeypatch):
    """Failure leg: if the reconnect flush raced the socket closing (send
    failed, pending kept), the stale snapshot that still arrives must NOT
    clear the pending style -- the NEXT successful reconnect has to
    deliver it.

    Seeded defect: add `self._pending_style = None` to _on_style_received
    (treating any incoming style as authoritative) -> pending lost -> red.
    """
    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    monkeypatch.setattr(
        tray_mod, "write_overlay_style", lambda style: None, raising=False,
    )
    tray, client = _tray(qapp)
    client.connected = False
    tray._send_style_change({**STYLE_DEFAULTS, "jp_size": 99})
    # Flush races the socket closing: send fails, pending must survive...
    tray._on_connected_changed(True)  # send_command still returns False
    # ...and the stale snapshot the server got out before the drop must
    # not cancel it.
    tray._on_style_received({"type": "style", **STYLE_DEFAULTS})
    assert tray._pending_style is not None and (
        tray._pending_style["jp_size"] == 99
    ), "stale on-connect snapshot cancelled the pending offline style"

    # Next successful reconnect delivers the flush.
    client.connected = True
    n = len(client.sent)
    tray._on_connected_changed(True)
    assert any(
        c.get("action") == "apply_style" and c["style"]["jp_size"] == 99
        for c in client.sent[n:]
    ), "pending style was not flushed on the next successful reconnect"


# --- Pause/genre sends while the WS is disconnected ---


def test_genre_pick_while_disconnected_flushes_on_reconnect(qapp):
    """A genre pick is durable user intent (the radio stays checked in
    the menu), so dropping it while the WS is down must not be a silent
    revert-on-reconnect -- it queues and flushes once the WS comes back,
    latest pick wins, one-shot.

    Seeded defect: revert _on_genre_chosen to the bare fire-and-forget
    send_command call -> nothing is flushed on reconnect, the server's
    state snapshot silently reverts the radio -> red.
    """
    tray, client = _tray(qapp)
    client.connected = False
    tray._on_genre_chosen("anime")
    tray._on_genre_chosen("drama")  # latest offline pick supersedes

    client.connected = True
    n_before = len(client.sent)
    tray._on_connected_changed(True)
    flushed = [c for c in client.sent[n_before:] if c.get("action") == "set_genre"]
    assert flushed == [
        {"type": "command", "action": "set_genre", "genre": "drama"}
    ], f"offline genre pick was not flushed (latest-wins) on reconnect; got {flushed}"

    # One-shot: a later reconnect must not replay it.
    n_after = len(client.sent)
    tray._on_connected_changed(True)
    assert all(
        c.get("action") != "set_genre" for c in client.sent[n_after:]
    ), "genre flush replayed on second connect"


def test_genre_flush_superseded_by_newer_online_pick(qapp):
    """A successful ONLINE genre send clears any genre still queued from
    an earlier offline pick -- otherwise the next reconnect would replay
    the stale pick over the newer one (same invariant as the offline
    style supersede).

    Seeded defect: drop the pending-clear from _on_genre_chosen's sent
    branch -> stale 'anime' replays after reconnect -> red.
    """
    tray, client = _tray(qapp)
    client.connected = False
    tray._on_genre_chosen("anime")
    client.connected = True
    tray._on_genre_chosen("news")  # online pick supersedes the queued one

    n_before = len(client.sent)
    tray._on_connected_changed(True)
    assert len(client.sent) == n_before, (
        "stale offline genre replayed after a newer online pick; the "
        "successful send must clear the pending genre"
    )


def test_pause_click_while_disconnected_is_dropped_not_queued(qapp):
    """Pause is a TRANSIENT runtime command, not durable intent -- when
    the WS is down the pipeline process is down/restarting, a fresh
    pipeline starts unpaused, and replaying a minutes-old pause on
    reconnect would silently mute subtitles with no fresh user action.
    A disconnected pause click is deliberately dropped: nothing is
    queued, and the menu label keeps following authoritative state
    broadcasts only (the on-connect state snapshot re-syncs it).

    Seeded defect: queue the dropped pause for a reconnect flush (the
    genre treatment) -> a pause command appears after reconnect -> red.
    """
    tray, client = _tray(qapp)
    client.connected = False
    tray._on_pause_toggled()
    # Label must not flip locally -- it follows pipeline state broadcasts.
    assert tray._pause_action.text() == "Pause"

    client.connected = True
    n_before = len(client.sent)
    tray._on_connected_changed(True)
    assert client.sent[n_before:] == [], (
        f"disconnected pause click was queued/replayed: {client.sent[n_before:]}"
    )


# --- the short menu and the flyout (phase 4) ---

_COMBOS = {"pause_resume": "Ctrl+Alt+P", "toggle_overlay": "Ctrl+Alt+O",
           "show_series": "Ctrl+Alt+S"}


def test_short_menu_is_pause_pick_settings_quit_with_the_live_combos(qapp, tmp_path, monkeypatch):
    """Right click (Decision 12): Pause or Resume, Pick a series..., Settings...,
    Quit Yamabiko, each bound key's combo after a tab and no QAction shortcut,
    kept current by state frames and by Settings saving new shortcuts, which
    also reach the flyout tooltips (Decisions 13 and 33).

    Seeded defects: keep the long menu, hard-code Ctrl+Alt+P, bind a QAction
    shortcut, or connect shortcutsChanged straight to _send_hotkeys (the
    labels never change) -> red."""
    # Isolate the real config the SettingsWindow reads to a tmp appdata (S4).
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, client = _tray(qapp, on_change_series=lambda: None, shortcuts=dict(_COMBOS))
    assert [a.text() for a in tray._menu.actions()] == [
        "Pause\tCtrl+Alt+P", "Pick a series\u2026\tCtrl+Alt+S", "",
        "Settings\u2026", "", "Quit Yamabiko",
    ]
    assert all(a.shortcut().isEmpty() for a in tray._menu.actions())
    tray._on_state("variety", True)
    assert tray._pause_action.text() == "Resume\tCtrl+Alt+P"
    tray._open_settings()
    win = tray._settings_window
    win.shortcutsChanged.emit({"pause_resume": "Ctrl+Alt+K", "show_series": "Ctrl+Alt+J"})
    assert tray._pause_action.text() == "Resume\tCtrl+Alt+K"
    assert tray._pick_series_action.text() == "Pick a series\u2026\tCtrl+Alt+J"
    assert tray._flyout._pause.toolTip() == "Resume (Ctrl+Alt+K)"
    assert client.sent[-1]["action"] == "set_hotkeys"
    win.close()


def test_with_shortcuts_off_the_menu_and_the_tooltips_show_no_combos(qapp, tmp_path, monkeypatch):
    """hotkeys.enabled false (shortcut_labels -> None, Decision 60): no combo
    in the menu or the flyout tooltips, and a Settings save adds none,
    because the pipeline binds nothing then.

    Seeded defect: treat None like an empty dict -> red."""
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, _ = _tray(qapp, shortcuts=None)
    tray._open_settings()
    win = tray._settings_window
    win.shortcutsChanged.emit({"pause_resume": "Ctrl+Alt+K"})
    assert tray._pause_action.text() == "Pause"
    assert tray._flyout._pause.toolTip() == "Pause"
    win.close()


def test_shortcut_labels_follow_the_merged_config():
    """Decisions 33 and 60: the stored spelling, a hand-written combo that
    doesn't parse shown as written, empty combos left out, no series combo
    with the series feature off, None with shortcuts off.

    Seeded defects: show the raw spelling, drop the feature gate, or return
    {} for enabled: false -> red."""
    from yamabiko.hotkeys import shortcut_labels

    cfg = {
        "gazetteer": {"enabled": True}, "romanization": {"enabled": True},
        "hotkeys": {"enabled": True, "pause_resume": "alt+ctrl+p",
                    "toggle_overlay": "Ctrl+Alt+Numpad5", "show_series": ""},
    }
    assert shortcut_labels(cfg) == {
        "pause_resume": "Ctrl+Alt+P", "toggle_overlay": "Ctrl+Alt+Numpad5",
    }
    cfg["hotkeys"]["show_series"] = "Ctrl+Alt+S"
    assert shortcut_labels(cfg)["show_series"] == "Ctrl+Alt+S"
    assert "show_series" not in shortcut_labels({**cfg, "gazetteer": {"enabled": False}})
    assert shortcut_labels({**cfg, "hotkeys": {**cfg["hotkeys"], "enabled": False}}) is None


def test_left_click_opens_the_flyout_and_the_click_that_closed_it_does_not_reopen_it(
    qapp, monkeypatch,
):
    """Decision 9: Trigger opens the flyout where tray_placement puts it. The
    flyout hides on the mouse-down of a click on the icon, so that click's
    Trigger (on mouse-up) is swallowed; the next click opens it; a double
    click always ends open; a right click pops the short menu up where
    tray_placement puts it and clears the swallow (Decision 63).

    Seeded defects: drop the activated connect, the dismissed connect, the
    DoubleClick branch, the Context branch or its popup -> red."""
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QSystemTrayIcon

    from yamabiko.overlay_app import tray_placement

    reason = QSystemTrayIcon.ActivationReason
    on_icon = [False]
    placed: list = []
    monkeypatch.setattr(tray_placement, "cursor_in", lambda rect: on_icon[0])
    monkeypatch.setattr(tray_placement, "place_flyout",
                        lambda rect, size: placed.append(size) or QPoint(40, 40))
    monkeypatch.setattr(tray_placement, "place_menu",
                        lambda rect, size: placed.append(size) or QPoint(60, 50))
    tray, _ = _tray(qapp, overlay_window=_FakeOverlay())
    flyout = tray._flyout
    try:
        tray._tray.activated.emit(reason.Trigger)
        assert flyout.isVisible() and flyout.pos() == QPoint(40, 40)
        assert placed == [flyout.size()]
        on_icon[0] = True
        flyout.hide()                                   # mouse-down on the icon
        tray._tray.activated.emit(reason.Trigger)       # ...its mouse-up
        assert not flyout.isVisible(), "the click that closed the flyout reopened it"
        tray._tray.activated.emit(reason.Trigger)
        assert flyout.isVisible(), "the next click did not open the flyout"
        flyout.hide()
        tray._tray.activated.emit(reason.DoubleClick)
        assert flyout.isVisible(), "a double click must end with the flyout open"
        flyout.hide()
        tray._tray.activated.emit(reason.Context)       # right click: the short menu
        assert tray._menu.isVisible() and tray._menu.pos() == QPoint(60, 50), (
            "the right click did not pop the short menu up where tray_placement puts it")
        assert placed[-1] == tray._menu.size(), "placed for another size than the menu shows at"
        tray._menu.hide()
        tray._tray.activated.emit(reason.Trigger)
        assert flyout.isVisible(), "a right click left the swallow armed"
    finally:
        tray._menu.hide()
        flyout.hide()


def test_flyout_buttons_send_todays_commands(qapp):
    """Decision 11: the big button pauses or resumes (its bool says which),
    Genre sends set_genre and Quit the graceful quit, as the menu did.

    Seeded defects: drop a connect, or send the opposite of the button -> red."""
    tray, client = _tray(qapp)
    flyout = tray._flyout
    flyout.pauseRequested.emit(True)
    flyout.pauseRequested.emit(False)
    flyout.genreChosen.emit("drama")
    flyout.quitRequested.emit()
    assert client.sent == [
        {"type": "command", "action": "pause"},
        {"type": "command", "action": "resume"},
        {"type": "command", "action": "set_genre", "genre": "drama"},
        {"type": "command", "action": "quit"},
    ]


def test_subtitles_tile_sends_toggle_overlay_and_toggles_the_window_offline(qapp):
    """Decision 11: the tile does what Ctrl+Alt+O does: the pipeline toggles
    every client; with the socket down only this window can follow.

    Seeded defects: also toggle the window when the send worked (the
    broadcast toggles it back), or do nothing offline -> red."""
    class _Window:
        toggles = 0

        def toggle_visible(self):
            self.toggles += 1

    window = _Window()
    tray, client = _tray(qapp, overlay_window=window)
    tray._flyout.subtitlesToggleRequested.emit()
    assert client.sent == [{"type": "command", "action": "toggle_overlay"}]
    assert window.toggles == 0
    client.connected = False
    tray._flyout.subtitlesToggleRequested.emit()
    assert window.toggles == 1


def test_mode_and_alignment_save_and_apply_like_a_settings_apply(qapp, monkeypatch, caplog):
    """Decisions 11 and 27: the Mode tile and the Alignment segments write
    overlay.style.<key> and apply it through the Settings Apply path
    (apply_style); a failed write still applies for the session; under
    --config (read-only) nothing is written.

    Seeded defects: skip the write, write under --config, or drop the apply
    when the write fails -> red."""
    import logging

    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    writes: list[dict] = []
    monkeypatch.setattr(tray_mod, "write_overlay_style", writes.append)
    tray, client = _tray(qapp, style_seed=dict(STYLE_DEFAULTS))
    tray._flyout.modeChosen.emit("learn")
    tray._flyout.alignmentChosen.emit("center")
    assert writes == [{"mode": "learn"}, {"alignment": "center"}]
    assert [(c["action"], c["style"]) for c in client.sent] == [
        ("apply_style", {"mode": "learn"}), ("apply_style", {"alignment": "center"}),
    ]
    assert tray._current_style["mode"] == "learn"

    def _denied(style):
        raise OSError(13, "Access is denied")

    monkeypatch.setattr(tray_mod, "write_overlay_style", _denied)
    with caplog.at_level(logging.WARNING, logger="yamabiko.overlay_app.tray"):
        tray._flyout.modeChosen.emit("watch")
    assert client.sent[-1]["style"] == {"mode": "watch"}
    assert "could not save overlay.style.mode" in caplog.text

    writes.clear()
    monkeypatch.setattr(tray_mod, "write_overlay_style", writes.append)
    read_only, ro_client = _tray(qapp, style_seed=dict(STYLE_DEFAULTS), settings_read_only=True)
    read_only._flyout.modeChosen.emit("learn")
    assert writes == [] and ro_client.sent[-1]["style"] == {"mode": "learn"}


def test_flyout_shows_the_state_style_series_and_window_visibility(qapp):
    """The flyout follows the real state: state frames (pause, genre), style
    frames (mode, alignment), series frames, and the overlay window's
    visibility (set_overlay_visible, wired to OverlayWindow.visibleChanged).
    Emitted through the client's real signals.

    Seeded defects: drop a setter call or a connect -> red."""
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    tray, client = _tray_with_real_signals(qapp, on_change_series=lambda: None)
    flyout = tray._flyout
    client.stateReceived.emit("anime", True)
    assert flyout._pause.text() == "Resume" and flyout._genre.value() == "anime"
    assert flyout._status.kind() == "paused"
    client.styleReceived.emit({"type": "style", **STYLE_DEFAULTS,
                               "mode": "learn", "alignment": "center"})
    assert flyout._mode.value() == "Learn" and flyout._alignment.value() == "center"
    client.seriesReceived.emit(154587, "Frieren (2023)", "葬送のフリーレン")
    assert flyout._series_text.title() == "Frieren"
    assert flyout._series_text.subtitle() == "葬送のフリーレン"
    client.stateReceived.emit("anime", False)
    tray.set_overlay_visible(False)
    assert flyout._subtitles.value() == "Hidden" and flyout._status.kind() == "hidden"
    client.seriesReceived.emit(None, "", "")
    assert flyout._series_text.title() == "No series" and flyout._series_text.muted()


def test_flyout_cover_comes_from_the_palette_and_follows_cover_changed(qapp, monkeypatch):
    """R3: the series row's cover is SeriesPalette.cover_for(media_id), and a
    coverChanged for the shown series replaces it; one for another series
    does not.

    Seeded defects: pass no cover, skip the coverChanged connect, or replace
    the cover for any media id -> red."""
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtGui import QColor

    from yamabiko.overlay_app.tray import Tray

    class _Palette(QObject):
        coverChanged = Signal(int)
        clearRequested = Signal()

        def __init__(self):
            super().__init__()
            self.covers: dict = {}

        def cover_for(self, media_id):
            return self.covers.get(media_id, QColor("#2b2b33"))

        def set_current_series(self, media_id, label=""):
            pass

        def apply_palette(self, palette):
            pass

    palette = _Palette()
    client = _make_signal_client_class()()
    tray = Tray(client, on_quit=lambda: None, on_change_series=lambda: None,
                series_palette=palette)
    series_calls: list = []
    cover_calls: list = []
    real_set_series, real_set_cover = tray._flyout.set_series, tray._flyout.set_cover
    monkeypatch.setattr(tray._flyout, "set_series",
                        lambda *a: (series_calls.append(a), real_set_series(*a)))
    monkeypatch.setattr(tray._flyout, "set_cover",
                        lambda c: (cover_calls.append(c), real_set_cover(c)))
    client.seriesReceived.emit(5, "Kusuriya (2023)", "")
    assert series_calls[-1] == ("Kusuriya (2023)", "", QColor("#2b2b33"))
    # TrayFlyout.set_series shows its cover through self.set_cover (the
    # recorder above); only the coverChanged replacements count below.
    cover_calls.clear()
    palette.covers[5] = QColor("#e4a128")
    palette.coverChanged.emit(6)
    assert cover_calls == []
    palette.coverChanged.emit(5)
    assert cover_calls == [QColor("#e4a128")]


def test_built_pick_saves_the_series_poster_and_clear_keeps_it(qapp, qtbot):
    """Phase-5 L11: when a pick's build succeeds the tray makes it the
    palette's active series, which saves its poster for the next boot; a
    later Clear leaves the file (a re-pick shows it at once).

    Seeded defects: _set_series never tells the palette -> no file -> red;
    Clear deletes the file -> red.
    """
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QColor, QPixmap

    from yamabiko.gazetteer.gazetteer import cover_path
    from yamabiko.overlay_app.series_palette import SeriesPalette

    poster = QPixmap(10, 14)
    poster.fill(QColor("#123456"))
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.WriteOnly)
    poster.save(buf, "PNG")
    buf.close()

    palette = SeriesPalette()
    qtbot.addWidget(palette)
    palette._on_cover_ready(5, bytes(data))  # the picked row's poster, decoded
    tray, _client = _tray(qapp, on_change_series=lambda: None, series_palette=palette)
    assert not cover_path(5).is_file()

    tray._on_gazetteer_built(5, "Kusuriya (2023)", "薬屋のひとりごと")
    assert cover_path(5).is_file()

    tray._on_series_clear()
    assert cover_path(5).is_file()


def test_flyout_clear_sends_set_series_without_a_series_and_queues_it_offline(
    qapp, tmp_path, monkeypatch,
):
    """Decisions 11 and 66: Clear sends set_series with a null media id; with
    the socket down it deletes active_series.json itself and flushes the
    clear once on reconnect. The row shows "No series" at once.

    Seeded defects: send a payload without media_id, skip the offline
    delete, or replay a clear as a pick -> red."""
    from yamabiko.gazetteer.gazetteer import save_active_series

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, client = _tray(qapp, on_change_series=lambda: None)
    tray._on_series_received(5, "Kusuriya (2023)", "")
    tray._flyout.clearSeriesRequested.emit()
    assert client.sent[-1] == {"type": "command", "action": "set_series", "media_id": None}
    assert tray._flyout._series_text.title() == "No series"

    save_active_series(5, "Kusuriya (2023)", None)
    client.connected = False
    tray._flyout.clearSeriesRequested.emit()
    assert not (tmp_path / "active_series.json").exists()
    client.connected = True
    n_before = len(client.sent)
    tray._on_connected_changed(True)
    tray._on_connected_changed(True)
    assert [c for c in client.sent[n_before:] if c.get("action") == "set_series"] == [
        {"type": "command", "action": "set_series", "media_id": None},
    ]


def test_icon_follows_pause_hidden_problems_the_taskbar_and_the_screen_scale(qapp, monkeypatch):
    """Decisions 14, 23 and 67: the icon is the painted state (paused wins
    over hidden), with the amber dot while a problem is active, its ring in
    the taskbar color, redrawn when the taskbar theme or the icon screen's
    scale changes; a frame that changes nothing visible sets no icon.

    Seeded defects: ignore the problem set, the taskbar watcher or the
    screen's logicalDotsPerInchChanged, or setIcon on every frame -> red."""
    from PySide6.QtGui import QImage

    from yamabiko import problems
    from yamabiko.overlay_app import tray_icons

    tray, _ = _tray(qapp)
    set_icons: list = []
    real_set_icon = tray._tray.setIcon
    monkeypatch.setattr(tray._tray, "setIcon",
                        lambda icon: (set_icons.append(icon), real_set_icon(icon)))

    def _shows(state, problem, taskbar):
        icon = tray._tray.icon()
        px = icon.availableSizes()[0].width()
        expected = tray_icons.paint_icon(px, state, problem, taskbar)
        return _icon_image(icon, px) == expected.convertToFormat(
            QImage.Format.Format_ARGB32_Premultiplied)

    theme = tray._taskbar.theme()
    assert _shows("listening", False, theme)
    tray._on_state("variety", True)
    assert _shows("paused", False, theme)
    tray._on_state("variety", True)   # the same frame again
    tray._on_state("anime", True)     # a genre change shows nothing on the icon
    assert len(set_icons) == 1, "setIcon ran for frames that changed nothing on the icon"
    tray._on_state("variety", False)
    tray.set_overlay_visible(False)
    assert _shows("hidden", False, theme)
    tray.set_problems([problems.audio_problem("Speakers")])
    assert _shows("hidden", True, theme)
    other = "light" if theme == "dark" else "dark"
    monkeypatch.setattr(tray._taskbar, "theme", lambda: other)
    tray._taskbar.changed.emit(other)
    assert _shows("hidden", True, other)
    monkeypatch.setattr(tray_icons, "tray_icon_px", lambda screen: 20)
    tray._icon_screen_watched.logicalDotsPerInchChanged.emit(120.0)
    assert tray._tray.icon().availableSizes()[0].width() == 20
    assert _shows("hidden", True, other)


def test_problem_start_toasts_with_the_app_icon_and_its_click_opens_the_flyout(qapp, monkeypatch):
    """Lead decision 5 and Decisions 24 and 26: one notification per problem
    start with the frame's title and body and the shipped app icon, no dot
    (screenshots 15/16); clicking it opens the flyout, whose header shows
    the problem title until the problem clears.

    Seeded defects: a painted or dotted icon for a problem, no click
    handler, or no set_problem -> red."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QIcon

    from yamabiko import paths, problems
    from yamabiko.overlay_app import tray_placement

    monkeypatch.setattr(tray_placement, "place_flyout", lambda rect, size: QPoint(0, 0))
    tray, _ = _tray(qapp, overlay_window=_FakeOverlay())
    shown: list[tuple] = []
    monkeypatch.setattr(tray._tray, "showMessage", lambda *a: shown.append(a))
    entry = problems.translator_problem(True)
    tray.set_problems([entry])
    tray.on_problem_started(entry)
    title, body, icon, _ms = shown[-1]
    assert (title, body) == ("Translations stopped",
                             "The translator isn't responding. Yamabiko is restarting it.")
    assert _icon_image(icon, 64) == _icon_image(QIcon(str(paths.app_icon_path())), 64)
    assert tray._flyout._status.kind() == "problem"
    assert tray._flyout._status.text() == "Translations stopped"
    tray._tray.messageClicked.emit()
    assert tray._flyout.isVisible(), "the problem notification's click did not open the flyout"
    tray._flyout.hide()
    tray.set_problems([])
    assert tray._flyout._status.kind() == "listening"


def test_flyout_changes_reach_an_open_settings_window_so_its_apply_keeps_them(
    qapp, tmp_path, monkeypatch,
):
    """Decision 68: a Mode change and a Position preset made in the flyout
    while Settings is open show in Settings and become its baseline, so
    Settings' Apply neither writes Mode back to the value it opened with nor
    writes the height the user had dragged before the preset.

    Seeded defects: drop the adopt call in _set_overlay_choice or in
    _on_position_preset -> Apply saves mode "watch" / v_offset_pct 40 -> red."""
    from PySide6.QtGui import QGuiApplication

    from yamabiko.config import load_full_config

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, _ = _tray(qapp, overlay_window=_FakeOverlay())
    tray._open_settings()
    win = tray._settings_window
    try:
        win._voffset.setValue(40)  # dragged in Settings, not applied yet
        tray._flyout.modeChosen.emit("learn")
        tray._flyout.positionPresetChosen.emit(QGuiApplication.primaryScreen().name(), 55.0)
        assert win._overlay_choices["mode"].value() == "learn"
        assert win._voffset.value() == 55
        assert win._apply()
        saved = load_full_config()["overlay"]
        assert saved["style"]["mode"] == "learn"
        assert saved["position"]["v_offset_pct"] != 40
    finally:
        win.close()


# --- Position (presets + placement mode) replaces Unlock ---


class _FakeOverlay:
    """Records set_position calls. The tray presets are the subject; a fake
    keeps the assertion on the exact dict handed to the window and avoids a
    real OverlayWindow firing write_overlay_position to disk.
    current_position feeds the flyout's Position picker on each open;
    banner_text lets Settings open over it."""

    def __init__(self):
        self.positions: list[dict] = []

    def set_position(self, pos: dict, *, persist: bool = True) -> None:
        self.positions.append(pos)

    def current_position(self) -> dict:
        # The flyout's Position picker reads it on every open.
        from yamabiko.overlay_app.position_config import DEFAULT_POSITION
        return dict(self.positions[-1]) if self.positions else dict(DEFAULT_POSITION)

    def banner_text(self) -> tuple[str, str]:
        # Settings' preview seeds its banner from the overlay (_open_settings).
        return ("バラエティ", "Variety")


def test_position_presets_send_bottom_and_middle_set_position_dicts(qapp):
    """The flyout's Position presets (positionPresetChosen) must hand
    OverlayWindow.set_position the exact anchor dict: Bottom -> v 88, Middle
    -> v 55, both at the default horizontal anchor of both alignments (Center
    h 50, Left 6). Emitted through the flyout's real signal so the connect is
    exercised too.

    Seeded defect: swap the Bottom/Middle v values (88<->55), change the
    centred h from 50, or drop the Left anchor -> the dict assertion goes
    red.
    """
    from PySide6.QtGui import QGuiApplication

    name = QGuiApplication.primaryScreen().name()
    fake = _FakeOverlay()
    tray, _ = _tray(qapp, overlay_window=fake)

    tray._flyout.positionPresetChosen.emit(name, 88.0)
    assert fake.positions[-1] == {
        "screen": name, "v_offset_pct": 88.0, "h_offset_pct": 50.0,
        "left_offset_pct": 6.0,
    }, f"Bottom preset dict wrong: {fake.positions[-1]}"

    tray._flyout.positionPresetChosen.emit(name, 55.0)
    assert fake.positions[-1] == {
        "screen": name, "v_offset_pct": 55.0, "h_offset_pct": 50.0,
        "left_offset_pct": 6.0,
    }, f"Middle preset dict wrong: {fake.positions[-1]}"


def test_unlock_overlay_action_and_lock_helpers_are_gone(qapp):
    """The interim Unlock overlay affordance is fully retired: no menu
    action carries the Lock/Unlock label and the tray no longer holds the
    lock helpers/state.

    Seeded defect: re-add self._lock_action = QAction("Unlock overlay", ...)
    -> the label reappears in the menu -> red.
    """
    tray, _ = _tray(qapp)
    labels = {a.text() for a in tray._menu.actions()}
    assert "Unlock overlay" not in labels and "Lock overlay" not in labels, (
        f"a Lock/Unlock action is still present: {sorted(t for t in labels if t)}"
    )
    assert not hasattr(tray, "_toggle_lock"), "_toggle_lock not removed"
    assert not hasattr(tray, "_lock_action"), "_lock_action not removed"
    assert not hasattr(tray, "_overlay_interactive"), (
        "_overlay_interactive shadow state not removed"
    )


def test_move_subtitles_action_constructs_and_enters_placement_controller(
    qapp, monkeypatch,
):
    """The flyout's Move subtitles... must construct a
    PlacementController bound to the overlay window with an on_save that
    saves through the window's set_position (its single persist path, off
    under --config), and enter() it (not merely construct it).

    Seeded defect: drop the self._placement.enter() call -> the fake
    controller's .active stays False -> red; pass the wrong window /
    callback -> the identity asserts go red.
    """
    from yamabiko.overlay_app import placement as placement_mod

    created: list = []

    class _FakeController:
        def __init__(self, window, on_save, palette=None, on_exit=None):
            self.window = window
            self.on_save = on_save
            self.palette = palette
            self.on_exit = on_exit
            self.active = False
            created.append(self)

        def enter(self):
            self.active = True

    monkeypatch.setattr(placement_mod, "PlacementController", _FakeController)
    fake = _FakeOverlay()
    tray, _ = _tray(qapp, overlay_window=fake)

    tray._flyout.moveSubtitlesRequested.emit()

    assert len(created) == 1, f"expected one controller; got {len(created)}"
    assert created[0].active is True, "Move subtitles constructed but never entered placement"
    assert created[0].window is fake, "controller not bound to the overlay window"
    pos = {"screen": "primary", "v_offset_pct": 70.0, "h_offset_pct": 50.0,
           "left_offset_pct": 6.0}
    created[0].on_save(pos)
    assert fake.positions[-1] == pos, (
        "controller on_save does not save through the overlay window's set_position"
    )
    assert created[0].palette is tray._palette, (
        "controller was not handed the tray's current theme palette; the "
        "placement hint would fall back to the dark default"
    )
    # == not is: tray._on_placement_exit is a bound method, freshly created on
    # each attribute access, so identity never holds; equality does (same
    # instance + same function).
    assert created[0].on_exit == tray._on_placement_exit, (
        "controller was not handed the tray's placement-exit callback; the "
        "settings live-preview sample would not be restored after placement "
        "(P1-14)"
    )


def test_move_subtitles_reentry_ignored_while_placement_active(qapp, monkeypatch):
    """A second Move subtitles click while a placement session is active must
    not spawn a second controller.

    Seeded defect: drop the `not self._placement.active` re-entry guard ->
    the second trigger builds another controller -> len(created) == 2 -> red.
    """
    from yamabiko.overlay_app import placement as placement_mod

    created: list = []

    class _FakeController:
        def __init__(self, window, on_save, palette=None, on_exit=None):
            self.active = False
            created.append(self)

        def enter(self):
            self.active = True

    monkeypatch.setattr(placement_mod, "PlacementController", _FakeController)
    fake = _FakeOverlay()
    tray, _ = _tray(qapp, overlay_window=fake)

    move = tray._flyout.moveSubtitlesRequested
    move.emit()
    move.emit()  # still active: must be ignored
    assert len(created) == 1, (
        f"re-click while active spawned a second controller: {len(created)}"
    )


def test_flyout_position_picker_is_rebuilt_from_the_live_screens_on_each_open(qapp, monkeypatch):
    """Decision 29 (was C4 for the old submenu): the Position picker's presets
    are rebuilt from the LIVE screen list every time the flyout opens, so a
    monitor hotplug after boot neither strands a dead screen nor misses a
    new one.

    Seeded defect: build the entries once in __init__ -> the unplug leaves
    four rows -> red."""
    from PySide6.QtCore import QPoint

    from yamabiko.overlay_app import tray as tray_mod

    screens = [("HOTPLUG-A", True), ("HOTPLUG-B", False)]
    monkeypatch.setattr(tray_mod, "live_screens", lambda: list(screens))
    monkeypatch.setattr(tray_mod.tray_placement, "place_flyout", lambda rect, size: QPoint(0, 0))
    tray, _ = _tray(qapp, overlay_window=_FakeOverlay())
    combo = tray._flyout._position

    def _rows():
        return [combo.itemText(i) for i in range(combo.count())]

    tray._show_flyout()
    assert _rows() == ["Bottom of Display 1", "Middle of Display 1",
                       "Bottom of Display 2", "Middle of Display 2"]
    tray._flyout.hide()
    screens[:] = [("HOTPLUG-B", True)]
    tray._show_flyout()
    assert _rows() == ["Bottom of Display 1", "Middle of Display 1"]
    tray._flyout.hide()


# --- Tooltip identity + diagnostics affordances ---
#
# The two Troubleshooting items below are emitted through the flyout's real
# signals, which both proves the connect(...) wiring AND invokes the handler
# body (docs/TESTING.md §3.9 Wired-not-exercised) -- a callable() check
# on the handler could not tell a real handler from a no-op lambda.


def _menu_action(tray, text: str):
    actions = {a.text(): a for a in tray._menu.actions()}
    assert text in actions, (
        f"tray menu has no {text!r} item; got {sorted(t for t in actions if t)}"
    )
    return actions[text]


def test_tooltip_is_the_status_sentence_with_hidden_and_problem_lines(qapp):
    """Decisions 15 and 61: line 1 "Yamabiko is listening (Anime: Frieren)"
    (the series title without its year; the genre alone without a series)
    or "Yamabiko is paused"; then "Subtitles hidden" and each active
    problem's title; never longer than Shell_NotifyIcon's 127 UTF-16 units
    (a title outside the BMP counts two per character), the series title
    shortened with an ellipsis; no version, no em dash.

    Seeded defects: keep the version tooltip, keep the year, drop a line,
    cut the problem lines instead of the title, or count code points
    instead of UTF-16 units -> red."""
    from yamabiko import problems

    tray, _ = _tray(qapp, on_change_series=lambda: None)
    tip = tray._tray.toolTip
    assert tip() == "Yamabiko is listening (Variety)"
    tray._on_state("anime", False)
    tray._on_series_received(154587, "Frieren (2023)", "葬送のフリーレン")
    assert tip() == "Yamabiko is listening (Anime: Frieren)"
    tray.set_overlay_visible(False)
    tray.set_problems([problems.translator_problem(True), problems.audio_problem("Speakers")])
    assert tip() == ("Yamabiko is listening (Anime: Frieren)\nSubtitles hidden\n"
                     "Translations stopped\nCan't hear your audio")
    tray._on_state("anime", True)
    assert tip().splitlines()[0] == "Yamabiko is paused"
    tray._on_state("anime", False)
    tray._on_series_received(1, "A" * 200 + " (2024)", "")
    lines = tip().splitlines()
    assert len(tip()) <= 127 and lines[0].endswith("\u2026)"), tip()
    assert lines[1:] == ["Subtitles hidden", "Translations stopped", "Can't hear your audio"]
    assert "\u2014" not in tip()
    tray._on_series_received(2, "\U0001D54F" * 200 + " (2024)", "")
    lines = tip().splitlines()
    assert len(tip().encode("utf-16-le")) // 2 <= 127 and lines[0].endswith("\u2026)"), tip()
    assert lines[1:] == ["Subtitles hidden", "Translations stopped", "Can't hear your audio"]


def test_open_log_folder_action_hands_logs_dir_to_startfile(
    qapp, tmp_path, monkeypatch,
):
    """The flyout's Troubleshooting > Open log folder invokes the handler,
    which opens paths.logs_dir() via os.startfile (patched -- we assert on
    the argument, the seam the OS actually sees).

    Seeded defects (one line each):
      - drop the openLogFolderRequested connect -> nothing
        recorded -> red.
      - startfile(paths.app_data_dir()) instead of logs_dir() -> the
        argument assertion goes red.
    """
    from pathlib import Path

    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    opened: list[Path] = []
    monkeypatch.setattr(
        "os.startfile", lambda p: opened.append(Path(p)), raising=False,
    )
    tray, _ = _tray(qapp)
    tray._flyout.openLogFolderRequested.emit()
    assert opened == [paths.logs_dir()], (
        f"'Open log folder' did not open the logs dir; startfile calls: {opened}"
    )


def test_save_diagnostics_collects_off_gui_thread_and_its_toast_shows_the_zip_in_explorer(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """Save diagnostics (flyout > Troubleshooting) must (1) return WITHOUT
    blocking the GUI thread on the collection -- the snapshot's
    nvidia-smi probe has a 5 s timeout and the zip deflate can add
    seconds more, which used to freeze the tray menu and overlay
    repaint -- (2) announce "Diagnostics saved" / "<file name>, on your
    desktop." with the shipped app icon once the worker reports back, and
    (3) a click on that notification opens Explorer with the zip selected
    (Decisions 16, 24 and 35).

    Seeded defects (one line each):
      - revert _on_collect_diagnostics to the synchronous
        `zip_path = collect_diagnostics()` call -> the collector runs
        on the GUI thread (worker-thread assert red) and the
        notification is already shown when trigger() returns
        (non-blocking assert red).
      - drop the showMessage from _on_diagnostics_done -> the final
        waitUntil times out -> red.
      - drop the on_click -> the click opens nothing -> red.
    """
    import threading

    from yamabiko.overlay_app import tray as tray_mod

    zip_path = tmp_path / "yamabiko-diagnostics-20260701-120000.zip"
    zip_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # minimal empty zip
    release = threading.Event()
    seen_threads: list[threading.Thread] = []

    def _slow_collect():
        seen_threads.append(threading.current_thread())
        release.wait(timeout=5)  # stand-in for the multi-second collection
        return zip_path

    monkeypatch.setattr(tray_mod, "collect_diagnostics", _slow_collect)
    tray, _ = _tray(qapp)
    shown: list[tuple] = []
    monkeypatch.setattr(
        tray._tray, "showMessage", lambda *a, **k: shown.append(a),
    )
    tray._flyout.saveDiagnosticsRequested.emit()
    # The click handler must have returned while the collection is
    # still in flight: nothing announced yet.
    assert shown == [], (
        "notification fired before the collection finished; the handler "
        "blocked the GUI thread on collect_diagnostics()"
    )
    qtbot.waitUntil(lambda: bool(seen_threads), timeout=2000)
    assert seen_threads[0] is not threading.main_thread(), (
        "collect_diagnostics ran on the GUI thread; a slow nvidia-smi "
        "probe would freeze the tray/overlay for seconds"
    )
    release.set()
    qtbot.waitUntil(lambda: bool(shown), timeout=2000)
    from PySide6.QtGui import QIcon

    from yamabiko import paths

    assert shown[0][:2] == ("Diagnostics saved", f"{zip_path.name}, on your desktop.")
    assert _icon_image(shown[0][2], 64) == _icon_image(QIcon(str(paths.app_icon_path())), 64)
    opened: list = []
    monkeypatch.setattr(tray_mod, "show_in_folder_outside_job", opened.append)
    tray._tray.messageClicked.emit()
    assert opened == [zip_path], f"the click did not show the zip in Explorer: {opened}"
    # Let the worker wind down before teardown destroys the Tray (a
    # QThread destructed while running aborts the process).
    qtbot.waitUntil(lambda: tray._diag_thread is None, timeout=2000)


def test_collect_diagnostics_action_ignores_reclick_while_collection_in_flight(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """A second menu click while a collection is still running must be
    ignored: two parallel workers would zip the same logs twice and
    fire two notifications for one user intent.

    Seeded defect: drop the isRunning() guard from
    _on_collect_diagnostics -> the second trigger starts a second
    worker -> the collector-call-count assertion goes red.
    """
    import threading

    from yamabiko.overlay_app import tray as tray_mod

    zip_path = tmp_path / "yamabiko-diagnostics-20260701-120000.zip"
    zip_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # minimal empty zip
    release = threading.Event()
    calls: list[threading.Thread] = []

    def _slow_collect():
        calls.append(threading.current_thread())
        release.wait(timeout=5)
        return zip_path

    monkeypatch.setattr(tray_mod, "collect_diagnostics", _slow_collect)
    tray, _ = _tray(qapp)
    shown: list[tuple] = []
    monkeypatch.setattr(
        tray._tray, "showMessage", lambda *a, **k: shown.append(a),
    )
    save = tray._flyout.saveDiagnosticsRequested
    save.emit()
    qtbot.waitUntil(lambda: bool(calls), timeout=2000)
    save.emit()  # re-click mid-flight: must be a no-op
    release.set()
    qtbot.waitUntil(lambda: bool(shown), timeout=2000)
    qtbot.waitUntil(lambda: tray._diag_thread is None, timeout=2000)
    qtbot.wait(200)  # drain: a stray second worker's signal would land here
    assert len(calls) == 1, "re-click started a second collection worker"
    assert len(shown) == 1, (
        f"re-click produced a second notification: {shown}"
    )


def test_collect_diagnostics_action_notifies_failure_when_no_zip_was_written(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """collect_diagnostics never raises; failure surfaces as a returned
    path that does not exist. The tray must then show a FAILURE
    notification pointing at the log folder -- not announce a zip that
    isn't there.

    Seeded defect: drop the exists() branch from _on_diagnostics_done
    and always announce success -> the failure-title assertion goes red.
    """
    from yamabiko.overlay_app import tray as tray_mod

    missing = tmp_path / "never-written.zip"
    monkeypatch.setattr(tray_mod, "collect_diagnostics", lambda: missing)
    tray, _ = _tray(qapp)
    shown: list[tuple] = []
    monkeypatch.setattr(
        tray._tray, "showMessage", lambda *a, **k: shown.append(a),
    )
    tray._flyout.saveDiagnosticsRequested.emit()
    qtbot.waitUntil(lambda: bool(shown), timeout=2000)
    assert "failed" in shown[0][0].lower(), (
        f"expected a failure notification title; got {shown[0]}"
    )
    assert str(missing) not in shown[0][1], (
        "failure notification announces a zip path that was never written"
    )
    assert "logs" in shown[0][1].lower(), (
        f"failure notification does not point at the log folder; got {shown[0]}"
    )
    qtbot.waitUntil(lambda: tray._diag_thread is None, timeout=2000)


def test_collect_diagnostics_action_notifies_failure_when_zip_write_fails_midway(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """END-TO-END failure leg through the REAL collector: a disk-full
    failure AFTER the zip was created (seeded ENOSPC on the snapshot
    writestr) must surface as the FAILURE notification, and no partial
    zip may be left behind to flip the tray's exists() check into
    announcing a corrupt half-bundle as success -- the exact bug the
    2026-07 review verified live.

    Seeded defect: drop the zip_path.unlink(missing_ok=True) from
    collect_diagnostics' outer except -> the partial zip survives ->
    the tray announces success -> the failure-title assertion goes red.
    """
    import zipfile

    from yamabiko import diagnostics

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "logs" / "main-overlay.log").write_text(
        "overlay line\n", encoding="utf-8",
    )
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    monkeypatch.setattr(diagnostics, "_desktop_dir", lambda: desktop)
    # Keep the unit tier fast/deterministic: skip the real nvidia-smi
    # probe inside the snapshot. The subject here is the failure-path
    # contract (partial-zip cleanup -> failure notification), not the
    # snapshot content, which tests/test_diagnostics.py owns.
    monkeypatch.setattr(diagnostics, "_snapshot_text", lambda: "snapshot\n")

    def _enospc_writestr(self, *args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(zipfile.ZipFile, "writestr", _enospc_writestr)

    tray, _ = _tray(qapp)
    shown: list[tuple] = []
    monkeypatch.setattr(
        tray._tray, "showMessage", lambda *a, **k: shown.append(a),
    )
    tray._flyout.saveDiagnosticsRequested.emit()
    qtbot.waitUntil(lambda: bool(shown), timeout=5000)
    assert "failed" in shown[0][0].lower(), (
        f"a partial zip was announced as success; got {shown[0]}"
    )
    assert list(desktop.glob("*.zip")) == [], (
        "partial diagnostics zip left on the Desktop after the failure"
    )
    qtbot.waitUntil(lambda: tray._diag_thread is None, timeout=2000)


def test_release_threads_detaches_running_diagnostics_worker(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """Quit while a diagnostics collection is in flight must DETACH the
    running QThread into thread_keeper (as release_threads already does for
    the gazetteer worker) instead of letting Tray teardown destroy it: a
    QThread destroyed while its OS thread runs is a Qt6 qFatal abort, and
    collect_diagnostics can block for seconds (the snapshot's nvidia-smi
    probe alone has a 5 s subprocess timeout). Its result slot must also be
    disconnected so no tray notification fires mid-teardown.

    Invocation-style (§3.9): drives a real _DiagnosticsThread to the running
    state, calls release_threads(), and asserts the observable detach +
    keeper state + slot disconnection -- not that a callable is registered.

    Seeded-defect drill: today release_threads ignores _diag_thread, so
    `thread.parent() is None` is false (still parented to the tray) -> red.
    Drop the finishedWithPath disconnect from release_threads -> the
    detached worker fires _on_diagnostics_done on completion -> the
    `shown == []` assertion goes red.
    """
    import threading

    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.ui import thread_keeper

    gate = threading.Event()
    zip_path = tmp_path / "yamabiko-diagnostics.zip"
    zip_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # minimal empty zip

    def _blocked_collect():
        gate.wait(timeout=10)  # stand-in for the multi-second collection
        return zip_path

    monkeypatch.setattr(tray_mod, "collect_diagnostics", _blocked_collect)
    tray, _ = _tray(qapp)
    shown: list[tuple] = []
    monkeypatch.setattr(tray._tray, "showMessage", lambda *a, **k: shown.append(a))

    tray._flyout.saveDiagnosticsRequested.emit()
    thread = tray._diag_thread
    qtbot.waitUntil(thread.isRunning, timeout=2000)

    tray.release_threads()

    try:
        assert thread.parent() is None, (
            "release_threads did not detach the running diagnostics worker; "
            "Tray teardown would destroy a running QThread (Qt6 qFatal)"
        )
        assert tray._diag_thread is None, (
            "release_threads left the _diag_thread reference dangling"
        )
        assert thread_keeper.any_running(), (
            "the detached diagnostics worker was not parked in thread_keeper"
        )
    finally:
        gate.set()
        qtbot.waitUntil(lambda: not thread.isRunning(), timeout=5000)
        thread_keeper.discard(thread)
    # The result slot was disconnected on release: the worker's completion
    # must not surface a tray notification during teardown.
    qtbot.wait(100)  # drain any queued finishedWithPath
    assert shown == [], (
        "a detached diagnostics worker still fired its tray notification "
        "(release_threads did not disconnect finishedWithPath)"
    )


def test_release_threads_detaches_active_quant_download_via_settings_window(
    qapp, qtbot, tmp_path,
):
    """Quit while the open Settings window is running a quant-tier model
    download must detach that download's worker into thread_keeper: the tray's
    release_threads (wired to app.aboutToQuit) reaches THROUGH the settings
    window to its in-flight download dialog, so a multi-GB pull blocked inside
    an uninterruptible hf_hub_download can't abort the process on quit.

    Invocation-style (§3.9): drives a real download worker to the running
    state, calls tray.release_threads(), asserts the observable park.

    Seeded-defect drill: drop the `win.release_threads()` call added to
    Tray.release_threads -> the running download thread is never parked ->
    the keeper assertion goes red.
    """
    import threading
    from pathlib import Path

    from yamabiko.overlay_app.model_fetch import QuantDownloadDialog, spec_for_tier
    from yamabiko.overlay_app.settings_window import SettingsWindow
    from yamabiko.ui import thread_keeper

    entered = threading.Event()
    gate = threading.Event()

    class _Blocking:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            entered.set()
            gate.wait(timeout=10.0)
            return Path("/fake/f")

    tray, _ = _tray(qapp)
    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
        quant_downloader=_Blocking(),
    )
    tray._settings_window = w
    dialog = QuantDownloadDialog(
        spec_for_tier("small"), "small", downloader=w._quant_downloader, parent=w,
    )
    w._active_quant_dialog = dialog
    dialog.start()
    assert entered.wait(5.0), "download worker never entered the blocking transfer"
    thread = dialog._thread
    try:
        tray.release_threads()
        assert any(t is thread for t, _ in thread_keeper._LINGERING), (
            "tray.release_threads did not reach the settings window's in-flight "
            "download; the running QThread would be destroyed on quit (Qt6 qFatal)"
        )
        assert thread_keeper.any_running()
    finally:
        gate.set()
        thread.wait(5000)
        thread_keeper.discard(thread)


def test_finished_diagnostics_thread_is_deleted_and_ref_cleared(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """A diagnostics worker that finishes normally must schedule its QThread
    wrapper for deleteLater and clear tray._diag_thread (mirroring
    _on_gaz_thread_finished). Otherwise finished wrappers accumulate as Tray
    children for the process lifetime, and -- as with the gazetteer worker --
    a re-click after a queued finish could read a deleted C++ object.

    Seeded-defect drill: drop the
    `thread.finished.connect(lambda t=thread: self._on_diag_thread_finished(t))`
    wiring from _on_collect_diagnostics -> _diag_thread never clears and the
    wrapper is never destroyed -> both waitUntils time out -> red. Revert the
    slot to an unconditional `self._diag_thread = None` without deleteLater ->
    the destroyed wait times out -> red.
    """
    from yamabiko.overlay_app import tray as tray_mod

    zip_path = tmp_path / "yamabiko-diagnostics.zip"
    zip_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # minimal empty zip
    monkeypatch.setattr(tray_mod, "collect_diagnostics", lambda: zip_path)
    tray, _ = _tray(qapp)
    monkeypatch.setattr(tray._tray, "showMessage", lambda *a, **k: None)

    tray._flyout.saveDiagnosticsRequested.emit()
    thread = tray._diag_thread
    destroyed: list[int] = []
    # destroyed fires only once the deferred deleteLater actually runs the
    # C++ destructor on a later event-loop turn -- an observable proxy for
    # "the finished wrapper was scheduled for destruction".
    thread.destroyed.connect(lambda *_: destroyed.append(1))

    qtbot.waitUntil(lambda: tray._diag_thread is None, timeout=3000)
    qtbot.waitUntil(lambda: bool(destroyed), timeout=3000)


# --- Settings window: single instance + live preview bridge (Task 10) ---


class _PreviewOverlay:
    """Records the overlay calls the Settings preview bridge makes:
    show_sample / hide_sample / apply_style, plus set_position -- the sink the
    Position page Apply (positionApplied) is wired to."""

    def __init__(self):
        self.samples: list[tuple] = []
        self.styles: list[dict] = []
        self.hidden = 0
        self.positions: list[dict] = []
        self.position_persists: list[bool] = []

    def show_sample(self, jp, en, jp_ruby=None):
        self.samples.append((jp, en))

    def hide_sample(self):
        self.hidden += 1

    def apply_style(self, style):
        self.styles.append(dict(style))

    def set_position(self, pos, *, persist=True):
        self.positions.append(dict(pos))
        self.position_persists.append(persist)

    def banner_text(self):
        return ("バラエティ", "Variety")


def test_style_preview_shows_sample_once_then_restyles(qapp):
    """_on_style_preview must show the sample subtitle exactly once (so a
    live control drag doesn't stack samples) and apply every preview style.

    Seeded defect: drop the `if not self._preview_active` guard in
    _on_style_preview -> the sample is shown per frame -> len(samples) == 2
    -> red.
    """
    ov = _PreviewOverlay()
    tray, _ = _tray(qapp, overlay_window=ov)
    tray._settings_window = _WindowSentinel()  # a window is open (belt precondition)
    tray._on_style_preview({"jp_size": 30})
    tray._on_style_preview({"jp_size": 40})
    assert len(ov.samples) == 1, (
        f"sample must show once, not per preview frame; got {ov.samples}"
    )
    assert ov.styles[-1] == {"jp_size": 40}
    assert tray._preview_active is True


def test_preview_ended_hides_sample_and_restores_committed_style(qapp):
    """_on_preview_ended must drop the sample and re-apply the committed
    style so an un-applied preview doesn't stick on the overlay.

    Seeded defect: drop the apply_style(self._current_style) restore ->
    the last preview style stays on screen -> the restore assertion red.
    """
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    ov = _PreviewOverlay()
    tray, _ = _tray(qapp, overlay_window=ov)
    tray._current_style = {**STYLE_DEFAULTS, "jp_size": 77}
    tray._settings_window = _WindowSentinel()  # a window is open (belt precondition)
    tray._on_style_preview({"jp_size": 30})
    tray._on_preview_ended()
    assert ov.hidden == 1
    assert ov.styles[-1]["jp_size"] == 77, (
        "committed style was not restored when the preview ended"
    )
    assert tray._preview_active is False
    assert tray._settings_window is None


def test_preview_ended_restores_seeded_style_not_factory_defaults(qapp):
    """P0-3 trigger (a): with the WS down, _current_style is seeded from the
    config-loaded overlay style (build_overlay passes overlay_cfg.style into
    style_seed). A preview then close must restore THAT seeded style -- not the
    factory STYLE_DEFAULTS, which would silently reset the user's configured
    jp_size until a WS style echo arrived (and never, while disconnected).

    Seeded defect: revert the ctor seed to `dict(STYLE_DEFAULTS)` (ignore
    style_seed) -> previewEnded restores jp_size 14, not 30 -> red.
    """
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    seed = {**STYLE_DEFAULTS, "jp_size": 30}
    ov = _PreviewOverlay()
    tray, _ = _tray(qapp, overlay_window=ov, style_seed=seed)
    tray._settings_window = _WindowSentinel()
    tray._on_style_preview({"jp_size": 48})  # user drags a control (preview runs)
    tray._on_preview_ended()                 # close without Apply (WS never echoed)
    assert ov.styles[-1]["jp_size"] == 30, (
        "previewEnded restored the factory default, clobbering the seeded "
        f"config style; got jp_size={ov.styles[-1]['jp_size']}"
    )


def test_preview_ended_after_connected_apply_restores_the_applied_style(qapp):
    """P0-3 trigger (b): a connected Apply/OK sends apply_style AND optimistically
    adopts it as _current_style, so the previewEnded fired by the closing window
    restores the JUST-APPLIED style -- not the pre-OK style for one server
    round-trip (or permanently if the broadcast echo is lost). Mirrors the real
    _ok() order: preview ran, then styleChanged -> _send_style_change, then
    close -> _on_preview_ended.

    Seeded defect: drop the optimistic `self._current_style = merged` from
    _send_style_change (keep only the offline branch's update) -> previewEnded
    restores the pre-OK jp_size 14, not 66 -> red.
    """
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    ov = _PreviewOverlay()
    tray, client = _tray(qapp, overlay_window=ov, style_seed=dict(STYLE_DEFAULTS))
    client.connected = True
    tray._settings_window = _WindowSentinel()
    tray._on_style_preview({"jp_size": 66})                     # preview ran
    tray._send_style_change({**STYLE_DEFAULTS, "jp_size": 66})  # OK -> apply_style
    tray._on_preview_ended()                                   # window closes
    assert any(
        c.get("action") == "apply_style" and c["style"]["jp_size"] == 66
        for c in client.sent
    ), "the connected Apply did not send apply_style"
    assert ov.styles[-1]["jp_size"] == 66, (
        "previewEnded reverted to the pre-OK style; the connected Apply must "
        f"optimistically update _current_style; got jp_size={ov.styles[-1]['jp_size']}"
    )


def test_preview_ended_without_a_preview_leaves_overlay_untouched(qapp):
    """A plain open+close of Settings with NO control change (no preview shown)
    must not reskin the live overlay: _on_preview_ended restores only when a
    preview actually ran this session (_preview_active). Otherwise the seed /
    committed style would be pushed onto the overlay on every close for no user
    action -- and (trigger a) a stale seed there would clobber the live style.

    Seeded defect: drop the `if self._preview_active` guard in
    _on_preview_ended -> apply_style + hide_sample run with no preview -> red.
    """
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    ov = _PreviewOverlay()
    tray, _ = _tray(qapp, overlay_window=ov, style_seed=dict(STYLE_DEFAULTS))
    tray._settings_window = _WindowSentinel()  # window opened...
    tray._on_preview_ended()                   # ...and closed with no control change
    assert ov.styles == [], (
        f"previewEnded reskinned the overlay though no preview ran; got {ov.styles}"
    )
    assert ov.hidden == 0, "previewEnded hid a sample that was never shown"
    assert tray._preview_active is False
    assert tray._settings_window is None


def test_style_preview_ignored_when_no_settings_window_is_open(qapp):
    """Belt (P2): a stylePreview that arrives with no settings window open -- a
    debounce frame that fired after the window closed -- must be a no-op, never
    resurrecting an orphaned sample on the live overlay. _on_style_preview
    no-ops when _settings_window is None.

    Seeded defect: drop the `if self._settings_window is None: return` belt in
    _on_style_preview -> the late frame shows a sample + applies the style ->
    red.
    """
    ov = _PreviewOverlay()
    tray, _ = _tray(qapp, overlay_window=ov)
    assert tray._settings_window is None
    tray._on_style_preview({"jp_size": 30})
    assert ov.samples == [], (
        f"a preview with no open settings window showed a sample; got {ov.samples}"
    )
    assert ov.styles == [], "a preview with no open settings window restyled the overlay"
    assert tray._preview_active is False


class _FakePlacement:
    """Minimal stand-in for PlacementController: only its ``active`` flag is read
    by the tray's sample-slot arbitration (_on_style_preview / _on_preview_ended
    check ``self._placement.active``)."""

    def __init__(self, active: bool = True):
        self.active = active


def test_style_preview_dropped_while_placement_active(qapp):
    """P1-14(b): a stylePreview frame arriving while an on-screen placement
    session is active must be DROPPED -- placement owns the single sample slot
    (the drag anchor), and showing the preview sample would replace it. The
    style is still cached so the preview can resume when placement exits.

    Seeded defect: drop the ``if self._placement is not None and ...active:
    return`` guard in _on_style_preview -> the preview sample replaces the drag
    anchor (ov.samples grows) and the overlay is restyled mid-drag -> red.
    """
    ov = _PreviewOverlay()
    tray, _ = _tray(qapp, overlay_window=ov)
    tray._settings_window = _WindowSentinel()
    tray._placement = _FakePlacement(active=True)

    tray._on_style_preview({"jp_size": 40})

    assert ov.samples == [], (
        "stylePreview replaced the placement drag anchor with the preview sample"
    )
    assert ov.styles == [], "stylePreview restyled the overlay during a live drag"
    assert tray._preview_style == {"jp_size": 40}, (
        "the dropped preview frame's style was not cached for restore-on-exit"
    )


def test_preview_ended_while_placement_active_keeps_drag_anchor(qapp):
    """P1-14(b): the settings window closing while placement is active must NOT
    hide the sample -- that slot is the placement drag anchor. previewEnded
    skips hide_sample while placement.active (placement teardown owns the
    sample) but still clears the preview + window refs.

    Seeded defect: drop the placement-active early-return in _on_preview_ended
    -> hide_sample runs and clears the drag anchor -> ov.hidden == 1 -> red.
    """
    ov = _PreviewOverlay()
    tray, _ = _tray(qapp, overlay_window=ov)
    tray._settings_window = _WindowSentinel()
    tray._preview_active = True  # a preview had run before placement started
    tray._placement = _FakePlacement(active=True)

    tray._on_preview_ended()

    assert ov.hidden == 0, "previewEnded hid the placement drag anchor"
    assert tray._preview_active is False
    assert tray._settings_window is None


def test_placement_exit_restores_borrowed_preview_sample(qapp):
    """P1-14(a) DRILL: after a preview ran, a placement session borrows the
    single sample slot and (on teardown) hides it. If the tray left
    _preview_active True with no sample on screen, every later stylePreview would
    no-op (``if not self._preview_active``) and the live preview would be dead
    for the rest of the settings session. The on_exit callback (_on_placement_exit)
    re-shows the preview sample and re-applies the LAST previewed style so the
    preview resumes.

    Seeded defect: replace the _on_placement_exit body with a bare
    ``self._preview_active = False`` (the pre-fix behaviour) -> no sample is
    re-shown (ov.samples stays length 1) and the restored style is missing -> red.
    """
    ov = _PreviewOverlay()
    tray, _ = _tray(qapp, overlay_window=ov)
    tray._settings_window = _WindowSentinel()

    # A style preview ran: sample shown once, _preview_active True, style cached.
    tray._on_style_preview({"jp_size": 40})
    assert len(ov.samples) == 1 and tray._preview_active is True

    # Placement borrowed + released the slot (its real teardown hid the sample);
    # on_exit fires to restore the settings preview.
    tray._on_placement_exit()

    assert len(ov.samples) == 2, (
        "placement exit did not re-show the borrowed preview sample; the live "
        "preview is dead for the rest of the settings session"
    )
    assert tray._preview_active is True, "preview flag lost across placement exit"
    assert ov.styles[-1] == {"jp_size": 40}, (
        "placement exit did not re-apply the LAST previewed style (it must use "
        "the cached _preview_style, not the committed _current_style)"
    )


def test_uncommitted_preview_restored_when_settings_closes_during_placement(qapp):
    """C3: interleaving preview -> placement -> settings window closed ->
    placement exits. The placement-active branch of _on_preview_ended cannot
    restore the committed style (placement owns the sample slot), and
    _on_placement_exit used to find the window ref already cleared and restore
    nothing -- the never-committed preview style stayed on the overlay for
    every subsequent subtitle. The close must leave a restore OWED, settled
    exactly once when the placement session tears down.

    Seeded defect: drop the _style_restore_owed bookkeeping -> after the
    placement exit the overlay's last applied style is still the jp_size 30
    preview -> red.
    """
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    ov = _PreviewOverlay()
    tray, _ = _tray(qapp, overlay_window=ov)
    tray._current_style = {**STYLE_DEFAULTS, "jp_size": 77}  # the committed style
    tray._settings_window = _WindowSentinel()

    tray._on_style_preview({"jp_size": 30})        # preview runs, never committed
    tray._placement = _FakePlacement(active=True)  # placement borrows the slot
    tray._on_preview_ended()                       # window closes mid-placement
    tray._placement.active = False                 # placement tears down...
    tray._on_placement_exit()                      # ...and fires on_exit

    assert ov.styles and ov.styles[-1]["jp_size"] == 77, (
        "the never-committed preview style stuck on the overlay after the "
        f"placement exit; applied styles: {ov.styles}"
    )
    # Settled exactly once: a later placement exit with nothing owed must not
    # re-apply the committed style again.
    n = len(ov.styles)
    tray._on_placement_exit()
    assert len(ov.styles) == n, "the owed restore was settled more than once"


def test_open_settings_wires_preview_signals_and_close(qapp, tmp_path, monkeypatch):
    """_open_settings must connect the SettingsWindow's stylePreview ->
    _on_style_preview and previewEnded -> _on_preview_ended. Proven by
    EMITTING through the real Signals on the constructed window (not calling
    the tray handlers directly), so a missing .connect goes red.

    Seeded defect: drop the win.stylePreview.connect(...) line -> the
    emission reaches no slot, no sample is shown -> red.
    """
    # Isolate the real config the SettingsWindow reads on construction
    # (load_full_config -> %LOCALAPPDATA%, incl. the legacy migration) to a
    # tmp appdata (S4).
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    ov = _PreviewOverlay()
    tray, _ = _tray(qapp, overlay_window=ov)
    tray._open_settings()
    win = tray._settings_window
    assert win is not None

    win.stylePreview.emit({"jp_size": 33})
    assert ov.samples, (
        "stylePreview emission did not reach the tray; the "
        "stylePreview.connect wiring is missing"
    )
    assert ov.styles[-1] == {"jp_size": 33}

    win.close()  # closeEvent -> previewEnded -> _on_preview_ended
    assert ov.hidden == 1, (
        "previewEnded emission did not reach the tray; the "
        "previewEnded.connect wiring is missing"
    )
    assert tray._settings_window is None


def test_open_settings_apply_emission_sends_apply_style(qapp, tmp_path, monkeypatch):
    """The styleChanged (Apply/OK) wiring must still reach _send_style_change,
    which forwards an apply_style command to the pipeline.

    Seeded defect: drop the win.styleChanged.connect(self._send_style_change)
    line -> no apply_style command is sent on Apply -> red.
    """
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    # Isolate the real config the SettingsWindow reads to a tmp appdata (S4).
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, client = _tray(qapp)
    tray._open_settings()
    win = tray._settings_window
    win.styleChanged.emit({**STYLE_DEFAULTS, "jp_size": 44})
    assert any(
        c.get("action") == "apply_style" and c["style"]["jp_size"] == 44
        for c in client.sent
    ), f"Apply did not forward an apply_style command; sent={client.sent}"
    win.close()


def test_open_settings_notice_emission_raises_tray_toast(
    qapp, tmp_path, monkeypatch,
):
    """_open_settings must wire the SettingsWindow's noticeRequested to a
    tray toast: on the OK flow the window closes immediately, so the toast
    is the only surface the deferred-recovery notice has left. Proven by
    EMITTING through the real Signal on the constructed window.

    Seeded defect: drop the win.noticeRequested.connect(...) line ->
    the emission reaches no slot, no toast -> red.
    """
    # Isolate the real config the SettingsWindow reads to a tmp appdata (S4).
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, _ = _tray(qapp)
    shown: list[tuple] = []
    monkeypatch.setattr(tray._tray, "showMessage", lambda *a, **k: shown.append(a))
    tray._open_settings()
    win = tray._settings_window
    win.noticeRequested.emit("Download deferred", "offered again next time")
    assert shown and shown[0][0] == "Download deferred" and (
        "offered again" in shown[0][1]
    ), f"noticeRequested emission did not reach the tray toast; shown={shown}"
    win.close()


def test_settings_shortcut_change_sends_set_hotkeys_and_replays_offline(qapp, tmp_path, monkeypatch):
    """Settings saved a new shortcut: the tray tells the pipeline to re-bind
    (set_hotkeys). Offline, it merges the changes and replays them once on
    reconnect.

    Seeded defects: never connect shortcutsChanged, drop the replay, keep
    only the latest offline change, never clear the pending change, or omit
    the replay flag -> red."""
    # Isolate the real config the SettingsWindow reads to a tmp appdata (S4).
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, client = _tray(qapp)
    tray._open_settings()
    win = tray._settings_window
    win.shortcutsChanged.emit({"pause_resume": "Ctrl+Alt+K"})
    assert client.sent[-1] == {"type": "command", "action": "set_hotkeys",
                               "hotkeys": {"pause_resume": "Ctrl+Alt+K"}}
    client.connected = False  # send_command now reports a dropped frame
    win.shortcutsChanged.emit({"toggle_overlay": "Ctrl+Alt+J"})
    win.shortcutsChanged.emit({"pause_resume": "Ctrl+Alt+L"})
    client.connected = True
    dropped = len(client.sent)  # the fake records the dropped frames too
    tray._on_connected_changed(True)
    assert client.sent[dropped:] == [{"type": "command", "action": "set_hotkeys",
                                      "hotkeys": {"toggle_overlay": "Ctrl+Alt+J",
                                                  "pause_resume": "Ctrl+Alt+L"},
                                      "replay": True}]
    replayed = len(client.sent)
    tray._on_connected_changed(True)
    assert len(client.sent) == replayed, "the replay was not one-shot"
    win.close()


def test_open_settings_at_a_page_opens_or_switches_the_window(qapp, tmp_path, monkeypatch):
    """The Shortcut unavailable notice opens Settings at Shortcuts, also when
    Settings is already open on another tab; the tray menu's Settings item
    opens it on the first page. Seeded defects: ignore the page for a new
    window, or for the open one, or disconnect the menu item -> red."""
    # Isolate the real config the SettingsWindow reads to a tmp appdata (S4).
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, _ = _tray(qapp)
    tray._open_settings(page="shortcuts")
    win = tray._settings_window
    assert win._stack.currentIndex() == 4
    win.show_page("timing")
    tray._open_settings(page="shortcuts")
    assert tray._settings_window is win and win._stack.currentIndex() == 4
    win.close()
    _menu_action(tray, "Settings\u2026").trigger()
    menu_win = tray._settings_window
    assert menu_win is not None and menu_win is not win
    assert menu_win._stack.currentIndex() == 0
    menu_win.close()


def test_shortcut_notice_click_opens_settings_at_shortcuts(qapp, tmp_path, monkeypatch):
    """A notice frame naming a Settings page makes its toast open Settings
    there when clicked; a notice without a page opens nothing. Emitted
    through the client's real noticeReceived signal. Seeded defects: show
    the toast without on_click, or with one for every notice -> red."""
    # Isolate the real config the SettingsWindow reads to a tmp appdata (S4).
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, client = _tray_with_real_signals(qapp)
    monkeypatch.setattr(tray._tray, "showMessage", lambda *a, **k: None)
    client.noticeReceived.emit("Download deferred", "offered again next time", "")
    tray._tray.messageClicked.emit()
    assert tray._settings_window is None, "a notice without a page opened Settings"
    client.noticeReceived.emit("Shortcut unavailable",
                               "Ctrl+Alt+S is used by another app. Pick a new one in Settings.",
                               "shortcuts")
    tray._tray.messageClicked.emit()
    assert tray._settings_window is not None and tray._settings_window._stack.currentIndex() == 4
    tray._settings_window.close()


def test_update_toast_click_handler_not_called_when_a_later_toast_is_clicked(
    qapp, qtbot, monkeypatch,
):
    """QSystemTrayIcon.messageClicked does not say which toast was clicked.
    The update toast used to keep its open-the-release-page handler
    connected until clicked, so after it timed out a click on ANY later
    toast (here the Settings notice) opened the GitHub release page. Every
    toast now goes through Tray.show_toast, which drops the previous
    toast's handler.

    Seeded defect: drop the _clear_toast_click_handler() call at the top of
    Tray.show_toast -> the stale update handler fires on the later click ->
    red.
    """
    import yamabiko.overlay_app.__main__ as overlay_main

    monkeypatch.setattr(
        overlay_main, "check_for_update",
        lambda current: ("9.9.9", "https://github.com/example/release-v9"),
    )
    opened: list[str] = []
    monkeypatch.setattr(overlay_main, "open_url_outside_job", opened.append)
    tray, _ = _tray(qapp)
    shown: list[tuple] = []
    monkeypatch.setattr(tray._tray, "showMessage", lambda *a, **k: shown.append(a))

    thread = overlay_main._schedule_update_check(
        tray, cfg={"updates": {"check_on_startup": True}},
    )
    try:
        qtbot.waitUntil(lambda: bool(shown), timeout=5000)
    finally:
        assert thread is not None and thread.wait(5000)
    assert shown[0][:2] == ("Yamabiko 9.9.9 is out", "Click to see what's new and download it.")

    tray._on_settings_notice("Download deferred", "offered again next time")
    tray._tray.messageClicked.emit()  # the user clicks the LATER toast

    assert [s[0] for s in shown] == ["Yamabiko 9.9.9 is out", "Download deferred"]
    assert opened == [], (
        f"a click on a later toast opened the update page: {opened}"
    )


def test_open_settings_position_apply_reaches_overlay_set_position(
    qapp, tmp_path, monkeypatch,
):
    """The Position page Apply must move the LIVE overlay: _open_settings wires
    the real SettingsWindow's positionApplied signal to
    OverlayWindow.set_position -- the SAME sink the tray presets use, which
    re-pins the overlay without persisting the committed position again. Proven by an
    actual Apply (a moved v-offset slider) on the constructed window, then
    asserting set_position received the anchor dict -- not by calling a tray
    handler directly, so a missing .connect goes red.

    Seeded defect (drill 2): drop the
    `win.positionApplied.connect(self._overlay_window.set_position)` line in
    _open_settings -> the Apply reaches no sink, ov.positions stays empty -> red.
    """
    # Isolate the real config the SettingsWindow reads/writes to a tmp appdata.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    ov = _PreviewOverlay()
    tray, _ = _tray(qapp, overlay_window=ov)
    tray._open_settings()
    win = tray._settings_window
    assert win is not None

    win._voffset.setValue(42)  # move the slider so the change-guard passes
    win._apply()

    assert ov.positions, (
        "positionApplied did not reach OverlayWindow.set_position; the "
        "positionApplied.connect wiring in _open_settings is missing"
    )
    assert ov.positions[-1]["v_offset_pct"] == 42.0, (
        f"the applied v did not reach set_position; got {ov.positions[-1]}"
    )
    assert ov.position_persists[-1] is False
    win.close()


def test_open_settings_single_instance_raises_existing(qapp, tmp_path, monkeypatch):
    """A second Settings click while a window is open must reuse (raise) it,
    not construct a duplicate -- and (S2) a MINIMIZED window must be restored:
    a minimized top-level still reports isVisible() True, and raise_/
    activateWindow do not clear WindowMinimized on Windows, so without
    showNormal() the "raise" leaves the window in the taskbar and the click
    looks dead.

    Seeded defects: drop the `existing.isVisible()` single-instance guard in
    _open_settings -> a second window is constructed and replaces the ref ->
    the identity assertion red; drop the isMinimized()/showNormal() branch ->
    the window stays minimized after the reopen click -> the not-minimized
    assertion red.
    """
    # Isolate the real config the SettingsWindow reads to a tmp appdata (S4).
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, _ = _tray(qapp)
    tray._open_settings()
    first = tray._settings_window
    assert first is not None and first.isVisible()
    tray._open_settings()
    assert tray._settings_window is first, (
        "second Settings click constructed a duplicate window instead of "
        "raising the existing one"
    )

    # S2: reopen from the tray while minimized must restore the window.
    first.showMinimized()
    assert first.isVisible() and first.isMinimized()  # the guard path runs
    tray._open_settings()
    assert tray._settings_window is first
    assert not first.isMinimized(), (
        "the Settings click left the existing window minimized; "
        "raise_/activateWindow alone cannot restore it on Windows"
    )
    first.close()


def test_settings_window_close_defers_final_ref_release_past_dispatch(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """C12: previewEnded is emitted from INSIDE SettingsWindow.closeEvent,
    before super().closeEvent returns. The tray holds the ONLY Python ref to
    the parentless (Python-owned) window in production, so nulling that ref
    synchronously in _on_preview_ended destroys the C++ widget while its own
    close dispatch is still unwinding -- the sporadic-crash class
    placement._teardown defers with singleShot(0). The tray must park the
    last ref past the dispatch: right after close() returns the C++ widget
    still exists, and it is released on a later event-loop turn (no leak).

    Mimics production: unlike its siblings, this test holds NO strong ref of
    its own across the close (a held ref would keep the widget alive and mask
    the bug).

    Seeded defect: revert _on_preview_ended to the bare
    `self._settings_window = None` -> the last ref drops inside the close
    dispatch, the widget is destroyed as soon as the borrowed close() handle
    unwinds -> the weakref is dead right after close() -> red.
    """
    import gc
    import weakref

    import shiboken6

    # Isolate the real config the SettingsWindow reads to a tmp appdata (S4).
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, _ = _tray(qapp)
    tray._open_settings()
    wr = weakref.ref(tray._settings_window)

    # Borrowed handle only: the evaluation temp is the sole extra ref and it
    # is gone once close() returns, exactly like the production close click.
    tray._settings_window.close()

    gc.collect()  # only a deliberately parked ref may keep the window alive
    obj = wr()
    assert obj is not None and shiboken6.isValid(obj), (
        "the settings window's C++ object did not survive its own closeEvent "
        "stack; the tray dropped the last Python ref mid-close-dispatch (C12)"
    )
    assert tray._settings_window is None, (
        "the tray still points at the closed window; the single-instance "
        "guard would raise a dead window"
    )
    del obj

    # ...and the parked ref is actually released on a later event-loop turn:
    # the window must not leak for the rest of the session.
    def _released() -> bool:
        gc.collect()
        o = wr()
        return o is None or not shiboken6.isValid(o)

    qtbot.waitUntil(_released, timeout=2000)


# --- Theme adoption: tray menu skin + live re-skin (Task 11) ---


@pytest.mark.parametrize("palette_name", ["night", "mist"])
def test_tray_menu_stylesheet_carries_palette_bg_hex(qapp, palette_name):
    """The tray QMenu must be skinned from the palette build_overlay hands
    it: the menu stylesheet carries that palette's raised hex (the
    QMenu background token).

    Seeded defect: drop the self._menu.setStyleSheet(menu_qss(self._palette))
    line in Tray.__init__ -> the menu stylesheet is empty, no raised hex -> red.
    """
    from yamabiko.overlay_app.theme import MIST, NIGHT

    palette = NIGHT if palette_name == "night" else MIST
    tray, _ = _tray(qapp, palette=palette)
    assert palette["raised"] in tray._menu.styleSheet(), (
        f"tray menu not skinned with the {palette_name} palette; "
        f"stylesheet={tray._menu.styleSheet()!r}"
    )


def test_apply_palette_swaps_tray_menu_stylesheet(qapp):
    """apply_palette re-skins the live tray menu: the old palette's raised hex is
    gone and the new one is present.

    Seeded-defect drill (b): make apply_palette skip the menu setStyleSheet
    -> the old night raised hex survives / the mist hex never appears -> red.
    """
    from yamabiko.overlay_app.theme import MIST, NIGHT

    tray, _ = _tray(qapp, palette=NIGHT)
    assert NIGHT["raised"] in tray._menu.styleSheet()

    tray.apply_palette(MIST)
    css = tray._menu.styleSheet()
    assert MIST["raised"] in css, (
        "apply_palette did not repaint the menu with the new palette"
    )
    assert NIGHT["raised"] not in css, (
        "apply_palette left the old palette hex in the menu stylesheet"
    )


def test_settings_theme_change_reskins_tray_menu(qapp, tmp_path, monkeypatch):
    """SettingsWindow.themeChanged must re-skin the tray menu through
    apply_palette: picking 'light' repaints the menu to the mist palette.
    Proven by EMITTING the real Signal on the constructed window (not calling
    the tray handler directly), so a missing .connect goes red.

    Seeded defect: drop the win.themeChanged.connect(...) line in
    _open_settings -> the emission reaches no slot, the menu keeps its night
    skin -> red.
    """
    from yamabiko.overlay_app.theme import MIST, NIGHT

    # Isolate the real config the SettingsWindow reads to a tmp appdata (S4).
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, _ = _tray(qapp, palette=NIGHT)
    tray._open_settings()
    win = tray._settings_window
    assert win is not None

    win.themeChanged.emit("light")
    css = tray._menu.styleSheet()
    assert MIST["raised"] in css and NIGHT["raised"] not in css, (
        "themeChanged emission did not re-skin the tray menu to the picked "
        f"theme; menu stylesheet={css!r}"
    )
    win.close()


def test_settings_theme_pick_does_not_reskin_originating_window_twice(
    qapp, tmp_path, monkeypatch,
):
    """K19: on a manual theme pick the settings window re-skins ITSELF
    (settings_window._on_theme_changed -> apply_palette) BEFORE emitting
    themeChanged. The tray's themeChanged handler must therefore re-skin only
    the REMAINING chrome (menu + active placement hint + series palette) and
    skip the originating window -- one repolish per surface per pick, not two
    full window repolishes. The OS colorSchemeChanged path (Tray.apply_palette)
    still reaches the window; that contract is covered by
    test_apply_palette_reskins_open_settings_window_from_given_palette.

    Seeded defect: revert _on_settings_theme_changed to the bare
    `self.apply_palette(palette_for(value))` -> the tray calls the window's
    apply_palette again -> the spy records the second skin -> red.
    """
    from yamabiko.overlay_app.theme import MIST, NIGHT

    # Isolate the real config the SettingsWindow reads to a tmp appdata (S4).
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, _ = _tray(qapp, palette=NIGHT)
    tray._open_settings()
    win = tray._settings_window
    assert win is not None

    # Spy on the tray-driven window re-skin (the window's own combo-driven
    # self-skin does not route through this instance attribute).
    tray_skins: list[dict] = []

    def _recording_apply(palette, _orig=win.apply_palette):
        tray_skins.append(palette)
        _orig(palette)

    win.apply_palette = _recording_apply

    # An active placement session records its own re-skin: the tray must
    # still forward the pick to the hint even while skipping the window.
    class _PaletteRecorder:
        active = True

        def __init__(self):
            self.applied: list[dict] = []

        def apply_palette(self, palette: dict) -> None:
            self.applied.append(palette)

    placement = _PaletteRecorder()
    tray._placement = placement

    win.themeChanged.emit("light")

    assert tray_skins == [], (
        "the tray re-skinned the originating settings window on its own "
        "themeChanged; the window already skinned itself (K19 double repolish)"
    )
    css = tray._menu.styleSheet()
    assert MIST["raised"] in css and NIGHT["raised"] not in css, (
        "the theme pick no longer re-skins the tray menu"
    )
    assert placement.applied and placement.applied[-1]["ground"] == MIST["ground"], (
        "the theme pick no longer reaches the active placement hint"
    )
    win.close()


def test_apply_palette_reskins_active_placement_hint(qapp, tmp_path, monkeypatch):
    """apply_palette forwards a live theme flip to the active placement hint:
    entering placement, then applying the mist palette, repaints the hint's
    raised token. Covers the tray -> placement forward the OS-follow path uses.

    Seeded defect: drop the `self._placement.apply_palette(palette)` forward in
    apply_palette -> the hint keeps the night raised hex -> red.
    """
    from yamabiko.overlay_app.overlay_window import OverlayWindow
    from yamabiko.overlay_app.theme import MIST, NIGHT

    # Isolate the overrides target (P2 test-config-write): this drives a REAL
    # PlacementController whose on_save is write_overlay_position, so a stray
    # save (e.g. a WindowDeactivate landing while the session is active) must
    # write to tmp_path, never the developer's real user_overrides.yaml.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    window = OverlayWindow()
    try:
        tray, _ = _tray(qapp, overlay_window=window, palette=NIGHT)
        tray._on_move_subtitles()  # constructs + enters a real PlacementController
        hint = tray._placement._hint
        assert NIGHT["raised"] in hint.styleSheet()

        tray.apply_palette(MIST)
        css = hint.styleSheet()
        assert MIST["raised"] in css and NIGHT["raised"] not in css, (
            f"apply_palette did not forward to the active placement hint; got {css!r}"
        )
    finally:
        if tray._placement is not None and tray._placement.active:
            tray._placement.cancel()
        window.close()


def test_apply_palette_reskins_open_settings_window_from_given_palette(
    qapp, tmp_path, monkeypatch,
):
    """apply_palette forwards a live theme flip to the OPEN settings window via
    its PUBLIC apply_palette, skinning it from the GIVEN palette -- NOT the
    window re-deriving from its own persisted theme. This is the tray ->
    settings-window forward the OS-follow path uses.

    The window is forced to boot dark (NIGHT) deterministically, then the tray
    applies MIST: the window's stylesheet must carry MIST's ground, not NIGHT's.

    Seeded defect (Finding 2): revert Tray.apply_palette to
    win.apply_palette(palette_for(win._theme_value)) -> the window re-derives
    NIGHT from its own 'dark' theme and ignores the passed MIST -> the MIST
    ground assertion red.
    """
    from yamabiko.overlay_app import settings_window as sw_mod
    from yamabiko.overlay_app.theme import MIST, NIGHT

    # Isolate the real config the SettingsWindow reads to a tmp appdata (S4).
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    # Force the window to boot dark independent of the CI host's real theme.
    monkeypatch.setattr(
        sw_mod.SettingsWindow, "_read_current_theme",
        staticmethod(lambda cfg=None: "dark"),
    )
    tray, _ = _tray(qapp, palette=NIGHT)
    tray._open_settings()
    win = tray._settings_window
    assert win is not None
    assert NIGHT["ground"] in win.styleSheet(), "settings window did not boot dark"

    tray.apply_palette(MIST)
    css = win.styleSheet()
    assert MIST["ground"] in css and NIGHT["ground"] not in css, (
        "tray.apply_palette did not re-skin the open settings window from the "
        f"GIVEN palette (the window re-derived its own theme): {css[:80]!r}"
    )
    win.close()


def test_apply_palette_survives_settings_window_reskin_failure(
    qapp, tmp_path, monkeypatch,
):
    """A theme flip is cosmetic and must never crash the overlay: if the open
    settings window's re-skin raises (e.g. a window mid-teardown whose C++
    object is gone), Tray.apply_palette still repaints the menu and returns
    normally -- the failure is logged, not propagated.

    Seeded defect (Finding 3): drop the try/except around the settings-window
    re-skin in apply_palette -> the RuntimeError propagates out of
    apply_palette -> red.
    """
    from yamabiko.overlay_app.theme import MIST, NIGHT

    # Isolate the real config the SettingsWindow reads to a tmp appdata (S4).
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, _ = _tray(qapp, palette=NIGHT)
    tray._open_settings()
    win = tray._settings_window
    assert win is not None

    def _boom(_palette):
        raise RuntimeError("C++ settings window already deleted")

    win.apply_palette = _boom

    # Must not raise despite the window re-skin blowing up:
    tray.apply_palette(MIST)

    # And the menu still got repainted with the new palette.
    assert MIST["raised"] in tray._menu.styleSheet(), (
        "apply_palette did not repaint the menu after the settings-window "
        "re-skin failed"
    )
    win.close()


def test_on_state_updates_the_pause_label_and_the_flyout_genre(qapp):
    """_on_state refreshes the menu and the flyout: pause action text + the
    flyout's Genre segment.

    Seeded-defect drill: stop setting the pause action text / the flyout
    genre in _on_state -> these assertions fail.
    """
    tray, _ = _tray(qapp)
    tray._on_state("anime", True)
    assert tray._pause_action.text() == "Resume"
    assert tray._flyout._genre.value() == "anime"

    tray._on_state("news", False)
    assert tray._pause_action.text() == "Pause"
    assert tray._flyout._genre.value() == "news"


# --- series selection: series row, offline queue, palette-driven build ---


def test_on_series_received_sets_and_clears_the_flyout_series_row(qapp):
    """The authoritative series broadcast from the pipeline drives the
    flyout's series row: a real media_id shows the title without its year,
    a None media_id shows "No series".

    Seeded-defect drill: stop calling _set_series in _on_series_received
    (or hardcode None) -> the row never shows the real series -> red.
    """
    # on_change_series set -> the series UI is built (the enabled config).
    tray, _ = _tray(qapp, on_change_series=lambda: None)
    tray._on_series_received(182205, "Slime (2024)")
    assert tray._flyout._series_text.title() == "Slime"

    tray._on_series_received(None, "")
    assert tray._flyout._series_text.title() == "No series"


def test_series_signal_emission_updates_the_flyout_series_row(qapp):
    """Emitting seriesReceived through a real Signal updates the flyout's
    series row — proving Tray.__init__ connected client.seriesReceived to
    _on_series_received (a freshly-started overlay learns the restored
    series this way on connect).

    Seeded defect: delete the `client.seriesReceived.connect(...)` line in
    tray.py -> the emission reaches no slot, the row stays 'No series' -> red.
    """
    # on_change_series set -> the series UI + seriesReceived listener are
    # wired (the enabled config; both are gated on the gazetteer feature).
    tray, client = _tray_with_real_signals(qapp, on_change_series=lambda: None)
    client.seriesReceived.emit(182205, "Slime (2024)", "")
    assert tray._flyout._series_text.title() == "Slime", (
        "seriesReceived emission did not reach _on_series_received; the "
        "seriesReceived.connect() wiring is missing"
    )


def test_gazetteer_built_online_sends_set_series_and_clears_pending(qapp):
    """Once the gazetteer cache is on disk, _on_gazetteer_built tells the
    pipeline via a set_series command (which loads the cache + hot-swaps +
    persists the pointer server-side) and updates the local label. A
    successful send leaves nothing queued.

    Seeded-defect drill: change the action string / drop a payload field ->
    the exact-payload assertion goes red; drop `self._pending_series = None`
    from the sent branch -> pending stays set -> red.
    """
    # on_change_series set -> the series UI is built so the label assertion
    # below is observable (the enabled config).
    tray, client = _tray(qapp, on_change_series=lambda: None)  # connected = True
    tray._on_gazetteer_built(182205, "Slime (2024)")
    # sent[0], not sent[-1]: the auto-genre ride-along (own tests below)
    # appends a set_genre after the set_series.
    assert client.sent[0] == {
        "type": "command", "action": "set_series",
        "media_id": 182205, "label": "Slime (2024)", "native": None,
    }
    assert tray._pending_series is None
    assert tray._flyout._series_text.title() == "Slime"


def test_gazetteer_built_while_disconnected_persists_pointer_and_queues(
    qapp, tmp_path, monkeypatch,
):
    """WS down (pipeline starting / reconnect backoff): the built gazetteer
    must not be a silent no-op. _on_gazetteer_built persists the
    active_series.json pointer (so a starting pipeline restores the choice)
    AND queues the pick for the reconnect flush.

    Seeded-defect drill: drop the save_active_series call -> the pointer file
    is absent -> red; drop `self._pending_series = (media_id, label)` -> no
    reconnect flush is queued -> red.
    """
    from yamabiko.gazetteer.gazetteer import load_active_series

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, client = _tray(qapp)
    client.connected = False

    tray._on_gazetteer_built(182205, "Slime (2024)")

    assert (tmp_path / "active_series.json").is_file(), (
        "offline series pick did not persist the active_series.json pointer"
    )
    assert load_active_series() == (182205, "Slime (2024)", None), (
        "persisted active-series pointer has the wrong shape/content"
    )
    assert tray._pending_series == (182205, "Slime (2024)", ""), (
        "offline series pick was not queued for the reconnect flush"
    )


def test_pending_series_flushes_to_pipeline_on_reconnect(qapp, tmp_path, monkeypatch):
    """The offline series pick reaches the pipeline once the WS comes back,
    or the server's on-connect state snapshot reverts the 'Series:' line.
    The flush fires once; a second connect must not re-send.

    Seeded-defect drill: drop the _pending_series flush block from
    _on_connected_changed -> nothing new is sent on reconnect -> red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, client = _tray(qapp)
    client.connected = False
    tray._on_gazetteer_built(182205, "Slime (2024)")
    n_before = len(client.sent)

    client.connected = True
    tray._on_connected_changed(True)

    flushed = [c for c in client.sent[n_before:] if c.get("action") == "set_series"]
    assert flushed == [{
        "type": "command", "action": "set_series",
        "media_id": 182205, "label": "Slime (2024)", "native": None,
    }], f"offline series pick was not flushed on reconnect; got {flushed}"
    assert tray._pending_series is None

    # One-shot: a later reconnect must not replay it.
    n_after = len(client.sent)
    tray._on_connected_changed(True)
    assert all(
        c.get("action") != "set_series" for c in client.sent[n_after:]
    ), "series flush replayed on second connect"


def test_series_pick_sends_native_title_with_set_series(qapp, tmp_path, monkeypatch):
    """A palette pick's Japanese title travels with set_series (online) and
    into the offline pointer.

    Seeded defect: drop native from the set_series payload -> red."""
    from yamabiko.gazetteer.gazetteer import load_active_series

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, client = _tray(qapp)
    tray._on_gazetteer_built(7, "Frieren (2023)", "葬送のフリーレン")
    sent = [c for c in client.sent if c.get("action") == "set_series"]
    assert sent and sent[-1]["native"] == "葬送のフリーレン", (
        f"set_series did not carry the Japanese title; sent={sent}"
    )

    client.connected = False
    tray._on_gazetteer_built(7, "Frieren (2023)", "葬送のフリーレン")
    assert load_active_series() == (7, "Frieren (2023)", "葬送のフリーレン"), (
        "the offline pointer did not keep the Japanese title"
    )


def test_settings_reduce_motion_apply_reaches_overlay_and_pipeline(
    qapp, tmp_path, monkeypatch,
):
    """Reduce motion applied in Settings turns the overlay's motion off at
    once and rides the apply_style command to the pipeline.

    Seeded defect: never connect reduceMotionChanged -> red."""
    from yamabiko.overlay_app import model_fetch
    from yamabiko.overlay_app.overlay_window import OverlayWindow

    # Isolate the real config the SettingsWindow reads/writes to a tmp appdata.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    monkeypatch.setattr(model_fetch, "tier_cached", lambda _tier: True)
    window = OverlayWindow()
    try:
        tray, client = _tray(qapp, overlay_window=window)
        tray._open_settings()
        win = tray._settings_window
        assert window.stack._reduce_motion is False  # precondition
        win._reduce_motion_cb.setChecked(True)
        assert win._apply() is True
        assert window.stack._reduce_motion is True, (
            "Settings reduce motion did not reach the live overlay"
        )
        styles = [c for c in client.sent if c.get("action") == "apply_style"]
        assert styles and styles[-1]["reduce_motion"] is True, (
            f"apply_style did not carry reduce_motion; sent={styles}"
        )
        win.close()
    finally:
        window.close()


def test_reduce_motion_from_settings_reaches_the_series_bar(qapp, qtbot):
    """L12: Reduce motion applied in Settings reaches the series bar at once,
    both ways (Tray._on_reduce_motion_changed, the slot Settings'
    reduceMotionChanged drives).

    Seeded defect: drop the series-bar forward -> the bar keeps animating ->
    red.
    """
    bar = _series_bar(qtbot, _apothecary_hit(), reduce_motion=False)
    tray, _client = _tray(qapp, on_change_series=bar.summon, series_palette=bar)
    tray._on_reduce_motion_changed(True)
    assert bar._reduce_motion is True, "Reduce motion on did not reach the series bar"
    tray._on_reduce_motion_changed(False)
    assert bar._reduce_motion is False, "Reduce motion off did not reach the series bar"


# --- auto-genre: a series pick rides a set_genre "anime" along ---


def test_series_pick_auto_sends_set_genre_anime_after_set_series(qapp):
    """With auto_genre on (the default), a series pick sends set_genre anime
    alongside set_series: the palette is AniList, which is anime-only, so the
    pick is durable "I am watching anime" intent. The genre rides the SAME
    persisted set_genre path as a manual tray pick, and a successful send
    leaves nothing queued.

    Seeded-defect drill: delete the auto-genre send block from
    _on_gazetteer_built -> only set_series is sent -> the actions list goes
    red on the missing second command.
    """
    tray, client = _tray(qapp)  # auto_genre defaults True; connected = True
    tray._on_gazetteer_built(4321, "Yani Neko")
    actions = [c["action"] for c in client.sent]
    assert actions == ["set_series", "set_genre"], (
        f"series pick did not ride a set_genre along; sent actions: {actions}"
    )
    assert client.sent[1] == {
        "type": "command", "action": "set_genre", "genre": "anime",
    }
    assert tray._pending_genre is None, (
        "a successful auto-genre send left a pending genre queued"
    )


def test_series_pick_with_auto_genre_off_sends_only_set_series(qapp):
    """auto_genre=False (the gazetteer.auto_genre toggle) must make the pick
    send set_series ONLY -- a user who turned the automation off keeps their
    manually chosen genre across picks.

    Seeded-defect drill: drop the `if self._auto_genre:` gate from
    _on_gazetteer_built (send unconditionally) -> a set_genre appears -> red.
    """
    tray, client = _tray(qapp, auto_genre=False)
    tray._on_gazetteer_built(4321, "Yani Neko")
    assert [c["action"] for c in client.sent] == ["set_series"], (
        f"auto_genre=False still sent a genre command: {client.sent}"
    )
    assert tray._pending_genre is None


def test_series_pick_while_disconnected_queues_genre_for_reconnect_flush(
    qapp, tmp_path, monkeypatch,
):
    """WS down: the auto-genre ride-along must not be a silent drop -- it
    queues on the EXISTING _pending_genre reconnect flush (the same queue a
    manual offline genre pick uses) and is delivered once the WS comes back.

    Seeded-defect drill: drop the `self._pending_genre = "anime"` from the
    auto-genre send-failed branch -> nothing queued, nothing flushed -> red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))  # offline pointer file
    tray, client = _tray(qapp)
    client.connected = False
    tray._on_gazetteer_built(4321, "Yani Neko")
    assert tray._pending_genre == "anime", (
        "offline series pick did not queue the auto genre for the reconnect flush"
    )

    client.connected = True
    n_before = len(client.sent)
    tray._on_connected_changed(True)
    flushed = [c for c in client.sent[n_before:] if c.get("action") == "set_genre"]
    assert flushed == [
        {"type": "command", "action": "set_genre", "genre": "anime"}
    ], f"queued auto genre was not flushed on reconnect; got {flushed}"


def _write_fresh_cache(media_id: int, mapping: dict) -> None:
    """A current-schema cache exactly as save() just wrote it (schema 2,
    built_at now): the pick-time freshness gate (cache_needs_rebuild)
    reads it as a plain cache hit, so the build worker touches no
    network. Legacy flat maps and >24h-old envelopes deliberately do NOT
    take that branch any more -- they rebuild (with the old file as the
    rebuild-failure fallback)."""
    from yamabiko.gazetteer.gazetteer import SeriesGazetteer, gazetteer_path

    path = gazetteer_path(media_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    SeriesGazetteer(dict(mapping)).save(path)


def _cached_series_pick_env(tmp_path, monkeypatch, media_id=182205):
    """Prime the REAL series-pick path for offline unit use: put a FRESH
    gazetteer cache for media_id on disk so the build worker takes its
    cache-hit branch (no network), and make AniListClient construction
    explode so a regression that reaches the network in a unit test is
    loud -- same discipline as
    test_on_series_chosen_builds_from_cache_and_sends_set_series."""
    from yamabiko.overlay_app import tray as tray_mod

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    def _no_network(*a, **k):
        raise AssertionError("worker hit AniList despite a cached gazetteer")

    monkeypatch.setattr(tray_mod, "AniListClient", _no_network)
    _write_fresh_cache(media_id, {"リムル": "Rimuru"})
    return media_id


def test_manual_genre_pick_during_build_window_is_not_clobbered_by_ride_along(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """A manual genre pick landing AFTER the palette pick but BEFORE the
    build's queued built signal is delivered (the AniList fetch is a
    multi-second window on a slow connection) is chronologically NEWER than
    the pick, so _on_gazetteer_built must skip the auto-genre ride-along:
    set_series still goes out, but no set_genre "anime". The invariant is
    "a manual genre change after a pick sticks until the next explicit pick",
    and the pick happens at palette-selection time, not build-completion time.

    Drives the REAL worker (cache hit, no network): on_series_chosen starts
    the thread, the manual pick lands before the event loop spins, then the
    queued built delivery arrives -- the production interleaving.

    Seeded-defect drill: gate the ride-along in _on_gazetteer_built on
    self._auto_genre alone (drop the pick-recency comparison) -> the built
    slot sends set_genre anime over the fresher manual drama -> the
    exactly-one-genre-command assertion goes red (this IS the pre-fix code;
    confirmed red against it).
    """
    media_id = _cached_series_pick_env(tmp_path, monkeypatch)
    tray, client = _tray(qapp)  # auto_genre defaults True; connected = True

    tray.on_series_chosen(media_id, "Slime (2024)")  # the pick
    tray._on_genre_chosen("drama")  # manual pick, AFTER the series pick
    qtbot.waitUntil(  # queued built delivery lands here
        lambda: any(c.get("action") == "set_series" for c in client.sent),
        timeout=3000,
    )
    qtbot.wait(100)  # drain finished->deleteLater + any stray set_genre

    genres = [c for c in client.sent if c.get("action") == "set_genre"]
    assert genres == [
        {"type": "command", "action": "set_genre", "genre": "drama"}
    ], (
        "the auto-genre ride-along clobbered a manual genre pick made during "
        f"the gazetteer build window; genre commands sent: {genres}"
    )


def test_manual_genre_pick_before_series_pick_still_gets_the_ride_along(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """The recency guard must key on the PICK, not on "any manual pick ever":
    a manual genre chosen BEFORE the series pick is superseded by the pick
    (the documented ride-along contract), so the built slot still sends
    set_genre anime. Guards the build-window fix against over-suppression.

    Seeded-defect drill: drop the `self._series_pick_genre_seq =
    self._manual_genre_seq` snapshot from on_series_chosen (leaving its
    construction-time value) -> the pre-pick manual drama makes the counters
    diverge -> the ride-along is skipped -> the trailing set_genre anime
    assertion goes red (drill run: seeded, watched it fail, restored).
    """
    media_id = _cached_series_pick_env(tmp_path, monkeypatch)
    tray, client = _tray(qapp)  # auto_genre defaults True; connected = True

    tray._on_genre_chosen("drama")  # manual pick, BEFORE the series pick
    tray.on_series_chosen(media_id, "Slime (2024)")  # the pick supersedes it
    qtbot.waitUntil(
        lambda: any(
            c.get("action") == "set_genre" and c.get("genre") == "anime"
            for c in client.sent
        ),
        timeout=3000,
    )
    actions = [c["action"] for c in client.sent]
    assert actions == ["set_genre", "set_series", "set_genre"], (
        f"expected manual drama, then the pick's set_series + anime "
        f"ride-along; sent actions: {actions}"
    )
    assert client.sent[-1] == {
        "type": "command", "action": "set_genre", "genre": "anime",
    }
    qtbot.wait(100)  # drain the worker's finished->deleteLater


def test_auto_genre_send_supersedes_stale_offline_genre_pick(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """A successful auto-genre send clears a genre still queued from an
    earlier offline MANUAL pick (same supersede invariant as
    _on_genre_chosen's sent branch) -- otherwise the next reconnect would
    replay the stale manual pick over the fresher auto 'anime'. Drives the
    REAL pick path (on_series_chosen, cache hit) because the ride-along only
    supersedes manual picks made BEFORE the series pick -- the pick-recency
    guard skips it otherwise -- and only the real path carries that ordering.

    Seeded-defect drill: drop `self._pending_genre = None` from the
    auto-genre sent branch -> the stale 'drama' replays after reconnect -> red.
    """
    media_id = _cached_series_pick_env(tmp_path, monkeypatch)
    tray, client = _tray(qapp)
    client.connected = False
    tray._on_genre_chosen("drama")  # queued offline, BEFORE the pick
    client.connected = True
    tray.on_series_chosen(media_id, "Slime (2024)")  # auto 'anime' rides online
    qtbot.waitUntil(
        lambda: any(
            c.get("action") == "set_genre" and c.get("genre") == "anime"
            for c in client.sent
        ),
        timeout=3000,
    )
    qtbot.wait(100)  # drain the worker's finished->deleteLater

    n_before = len(client.sent)
    tray._on_connected_changed(True)
    assert all(
        c.get("action") != "set_genre" for c in client.sent[n_before:]
    ), (
        "stale offline genre replayed after a successful auto-genre send; "
        f"got {client.sent[n_before:]}"
    )


def test_set_auto_genre_flips_the_ride_along_live(qapp):
    """set_auto_genre (the settings-checkbox sink) must take effect on the
    NEXT pick, both directions: off -> no set_genre, back on -> set_genre
    rides again.

    Seeded-defect drill: make set_auto_genre a no-op (drop the assignment) ->
    the first pick after set_auto_genre(False) still sends set_genre -> red.
    """
    tray, client = _tray(qapp)
    tray.set_auto_genre(False)
    tray._on_gazetteer_built(4321, "Yani Neko")
    assert [c["action"] for c in client.sent] == ["set_series"], (
        f"set_auto_genre(False) did not stop the ride-along: {client.sent}"
    )

    tray.set_auto_genre(True)
    n = len(client.sent)
    tray._on_gazetteer_built(4321, "Yani Neko")
    assert [c["action"] for c in client.sent[n:]] == ["set_series", "set_genre"], (
        f"set_auto_genre(True) did not re-enable the ride-along: {client.sent[n:]}"
    )


def test_open_settings_auto_genre_toggle_reaches_the_next_series_pick(
    qapp, tmp_path, monkeypatch,
):
    """_open_settings must connect the SettingsWindow's autoGenreChanged ->
    set_auto_genre: unchecking the REAL checkbox flips the tray's flag so the
    NEXT series pick sends no set_genre. Driven by toggling the constructed
    window's checkbox (not calling the tray setter directly), so a missing
    .connect goes red.

    Seeded-defect drill: drop the
    `win.autoGenreChanged.connect(self.set_auto_genre)` line in _open_settings
    -> the toggle reaches no slot, the pick still rides set_genre -> red.
    """
    # Isolate the real config the SettingsWindow reads/writes to a tmp appdata.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tray, client = _tray(qapp)
    tray._open_settings()
    win = tray._settings_window
    assert win is not None

    from yamabiko.overlay_app import model_fetch
    monkeypatch.setattr(model_fetch, "tier_cached", lambda _tier: True)
    win._auto_genre_cb.setChecked(False)
    assert win._apply() is True
    tray._on_gazetteer_built(4321, "Yani Neko")
    assert all(c.get("action") != "set_genre" for c in client.sent), (
        "unchecking the settings auto-genre checkbox did not reach the tray; "
        f"the autoGenreChanged.connect wiring is missing. Sent: {client.sent}"
    )
    win.close()


def test_pick_a_series_and_the_flyout_change_summon_the_palette(qapp):
    """The short menu's 'Pick a series…' and the flyout's Change… invoke the
    injected on_change_series callback (palette.summon in production).
    Driven via QAction.trigger() and the flyout's real signal so it proves
    BOTH the connect wiring and the handler body (§3.9 wired-not-exercised).

    Seeded-defect drill: drop the triggered.connect(self._on_change_series_clicked)
    line or the changeSeriesRequested connect -> that summon is missing ->
    red; make _on_change_series_clicked a no-op -> same.
    """
    summons: list[bool] = []
    tray, _ = _tray(qapp, on_change_series=lambda: summons.append(True))
    _menu_action(tray, "Pick a series\u2026").trigger()
    tray._flyout.changeSeriesRequested.emit()
    assert summons == [True, True], (
        "'Pick a series…' or Change… did not summon the palette via on_change_series"
    )


def test_change_series_clicked_is_safe_without_a_callback(qapp):
    """A tray built without an on_change_series callback (unit context) must
    not crash when the item is clicked.

    Seeded-defect drill: remove the `is not None` guard in
    _on_change_series_clicked -> None() raises TypeError -> red.
    """
    tray, _ = _tray(qapp)  # on_change_series defaults to None
    tray._on_change_series_clicked()  # must not raise


def test_tray_omits_series_ui_when_gazetteer_disabled(qapp):
    """No on_change_series callback (build_overlay's signal that the gazetteer
    feature is off) means the tray builds NO Pick a series... item and NO
    flyout series row, and does NOT connect seriesReceived.

    Seeded-defect drill: drop the `if on_change_series is not None:` gate on
    Pick a series... -> red; pass show_series=True to the flyout -> red.
    """
    tray, client = _tray_with_real_signals(qapp)  # on_change_series=None

    assert tray._pick_series_action is None, (
        "the tray built Pick a series... with no on_change_series callback"
    )
    texts = [a.text() for a in tray._menu.actions()]
    assert not any(t.startswith("Pick a series") for t in texts), texts
    assert tray._flyout._series_row is None, "the flyout built a series row"
    # An authoritative seriesReceived frame (never connected) is inert.
    client.seriesReceived.emit(182205, "Slime (2024)", "")
    assert tray._flyout._series_row is None


def _write_profile_stub(media_id: int) -> None:
    """Minimal valid profile-<media_id>.json beside the gazetteer cache: the
    both-present on-disk state a completed full pick leaves behind. Cache-hit
    tests that pin ZERO network must write this too -- a gazetteer cache with
    NO profile beside it is the v1-era state, which legitimately spends one
    intel request on the profile backfill."""
    import json

    from yamabiko.gazetteer.series_profile import profile_path

    p = profile_path(media_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "media_id": media_id, "genres": [], "tags": [],
        "synopsis_first_sentence": "", "relations": [],
    }), encoding="utf-8")


def test_on_series_chosen_builds_from_cache_and_sends_set_series(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """A palette selection drives the real build worker: on_series_chosen
    starts _GazetteerBuildThread, which (finding a FRESH gazetteer cache --
    current schema, built inside the 24 h horizon -- AND its profile on
    disk) skips the network entirely and emits built, whose queued slot
    sends the set_series command. Drives the live QThread to completion.

    The AniListClient is patched to explode so this proves the CACHE-HIT path
    is taken (never a silent unit-test network call): a regression that
    drops the cache_needs_rebuild gate the wrong way would hit the exploding
    client, emit buildFailed instead, and the set_series wait would time out.

    Seeded-defect drill: drop `thread.built.connect(self._on_gazetteer_built)`
    from on_series_chosen -> the build finishes but nothing is sent -> the
    waitUntil times out -> red.
    """
    from yamabiko.overlay_app import tray as tray_mod

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    _write_fresh_cache(182205, {"リムル": "Rimuru"})
    _write_profile_stub(182205)  # both present -> zero requests

    def _no_network(*a, **k):
        raise AssertionError("worker hit AniList despite a cached gazetteer")

    monkeypatch.setattr(tray_mod, "AniListClient", _no_network)

    tray, client = _tray(qapp)  # connected = True
    tray.on_series_chosen(182205, "Slime (2024)")

    qtbot.waitUntil(
        lambda: any(c.get("action") == "set_series" for c in client.sent),
        timeout=3000,
    )
    sent = [c for c in client.sent if c.get("action") == "set_series"]
    assert sent[-1] == {
        "type": "command", "action": "set_series",
        "media_id": 182205, "label": "Slime (2024)", "native": None,
    }
    # Drain: the worker's finished->deleteLater fires on the event loop
    # (the set_series wait above already spun it), so the QThread is torn
    # down cleanly. Do NOT touch tray._gaz_thread here -- deleteLater has
    # already deleted the C++ object.
    qtbot.wait(100)


def test_on_series_chosen_build_failure_notifies_and_switches_nothing(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """When the AniList fetch/build fails (no cache + network down), the
    worker emits buildFailed; the queued slot shows a 'Series not changed'
    warning and NO set_series command is sent -- the series must not switch
    to a hint-less state silently.

    Seeded-defect drill: drop
    `thread.buildFailed.connect(self._on_gazetteer_build_failed)` from
    on_series_chosen -> no notification -> the shown wait times out -> red;
    make the worker emit built on failure -> a set_series slips through -> the
    "no set_series" assertion goes red.
    """
    from yamabiko.overlay_app import tray as tray_mod

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))  # empty -> cache miss

    class _FailingClient:
        # Network-down on EVERY wire call the build makes: the full pick
        # build asks for the intel first, then the characters.
        def fetch_media_intel(self, media_id):
            raise RuntimeError("network down")

        def fetch_characters(self, media_id, max_pages: int = 6, on_page=None):
            raise RuntimeError("network down")

        def close(self):
            pass

    monkeypatch.setattr(tray_mod, "AniListClient", lambda *a, **k: _FailingClient())

    tray, client = _tray(qapp)
    shown: list[tuple] = []
    monkeypatch.setattr(tray._tray, "showMessage", lambda *a, **k: shown.append(a))

    tray.on_series_chosen(99999, "Nonexistent Series")

    qtbot.waitUntil(lambda: bool(shown), timeout=3000)
    assert "not changed" in shown[0][0].lower(), (
        f"expected a 'Series not changed' notification; got {shown[0]}"
    )
    qtbot.wait(100)  # drain: a stray built->set_series would land here
    assert all(c.get("action") != "set_series" for c in client.sent), (
        f"a failed gazetteer build still switched the series: {client.sent}"
    )
    # finished->deleteLater already tore the worker down on the event loop;
    # don't touch tray._gaz_thread (its C++ object is gone).
    qtbot.wait(50)


def test_on_series_chosen_empty_fetch_does_not_poison_cache(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """A genuine empty-200 from AniList (the series has no characters yet) must
    NOT write an empty {} gazetteer cache: a cached empty file makes
    path.is_file() short-circuit every later pick, so the fetch never retries.
    The build still succeeds ('no hints yet' is a valid outcome), so set_series
    is still sent -- but no poisoned cache file is left behind.

    Seeded-defect drill: revert the `if gaz.map:` guard in
    _GazetteerBuildThread.run to an unconditional `gaz.save(path)` -> an
    empty {} cache file is written -> the 'cache absent' assertion goes red.
    """
    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.gazetteer.gazetteer import gazetteer_path

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))  # empty -> cache miss

    class _EmptyClient:
        # A series page with nothing on it: empty intel AND an empty cast
        # (the full pick build fetches both).
        def fetch_media_intel(self, media_id):
            return {"genres": [], "tags": [], "description": "",
                    "relations": []}

        def fetch_characters(self, media_id, max_pages: int = 6, on_page=None):
            return []

        def close(self):
            pass

    monkeypatch.setattr(tray_mod, "AniListClient", lambda *a, **k: _EmptyClient())

    tray, client = _tray(qapp)  # connected = True
    tray.on_series_chosen(182205, "Slime (2024)")

    # The build succeeds ('no hints yet'): set_series is still sent.
    qtbot.waitUntil(
        lambda: any(c.get("action") == "set_series" for c in client.sent),
        timeout=3000,
    )
    assert not gazetteer_path(182205).is_file(), (
        "an empty fetch wrote a gazetteer cache file; a re-pick would "
        "short-circuit on path.is_file() and never retry the fetch"
    )
    qtbot.wait(100)  # drain the worker's finished->deleteLater


def test_build_worker_reports_staged_progress_for_its_media_id(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """Phase-5 L10: a cache-miss build forwards build_series_gazetteer's
    staged fractions as progress(media_id, fraction) before built, so the
    series bar's loading line follows the real requests.

    Seeded defects: build without on_progress -> no progress -> red; emit
    another media id -> red; report progress after built -> red.
    """
    from yamabiko.gazetteer.anilist import Character
    from yamabiko.overlay_app import tray as tray_mod

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))  # empty -> cache miss

    class _OnePageClient:
        def fetch_media_intel(self, media_id):
            return {"genres": [], "tags": [], "description": "", "relations": []}

        def fetch_characters(self, media_id, max_pages: int = 6, on_page=None):
            if on_page is not None:
                on_page(1.0)
            return [Character("ヤニ", "Yani", [], role="MAIN")]

        def close(self):
            pass

    monkeypatch.setattr(tray_mod, "AniListClient", lambda *a, **k: _OnePageClient())
    thread = tray_mod._GazetteerBuildThread(207141, "Yani Neko (2026)")
    seen: list[tuple] = []
    thread.progress.connect(lambda mid, fraction: seen.append((mid, fraction)))
    thread.built.connect(lambda *_args: seen.append(("built", None)))
    with qtbot.waitSignal(thread.built, timeout=3000):
        thread.start()
    assert thread.wait(3000)
    qtbot.wait(100)  # the worker has ended: deliver every report it queued
    assert seen[-1] == ("built", None), f"progress arrived after built: {seen}"
    assert [mid for mid, _fraction in seen[:-1]] == [207141, 207141]
    assert [fraction for _mid, fraction in seen[:-1]] == pytest.approx([0.15, 0.75])


class _IntelCarryClient:
    """Faithful AniListClient stand-in for the FULL pick-time build:
    fetch_media_intel serves the flattened intel shape (one PREQUEL ANIME
    relation) and fetch_characters serves distinct casts for the picked and
    the related media. Only build_series_gazetteer ever asks for the intel
    or the related cast -- the old direct fetch_characters+from_characters
    path can reach neither, which is exactly what the two tests below pin.
    ``closed`` counts close() calls so the worker's per-pick client hygiene
    (close in the finally) is assertable."""

    PICKED = 207141
    RELATED = 41042

    def __init__(self):
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    def fetch_media_intel(self, media_id: int) -> dict:
        assert media_id == self.PICKED, (
            f"intel fetched for media {media_id}, not the picked series"
        )
        return {
            "genres": ["Comedy"],
            "tags": [],
            "description": "A cat takes up smoking. Chaos follows.",
            "relations": [
                {"relationType": "PREQUEL", "id": self.RELATED,
                 "type": "ANIME", "format": "TV", "title": "Yani Neko Zero"},
            ],
        }

    def fetch_characters(self, media_id: int, max_pages: int = 6, on_page=None):
        from yamabiko.gazetteer.anilist import Character
        if media_id == self.PICKED:
            return [Character("ヤニねこ", "Cigarette Cat", [], role="MAIN")]
        if media_id == self.RELATED:
            return [Character("ゼロねこ", "Zero Cat", [], role="MAIN")]
        raise AssertionError(
            f"characters fetched for unexpected media {media_id}"
        )


def test_on_series_chosen_carries_related_media_characters_into_the_cache(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """A cache-miss palette pick builds the on-disk gazetteer through
    build_series_gazetteer: the picked media's characters PLUS the relations
    carry (a PREQUEL's cast keeps talking in the picked season). The cached
    map must contain a related-media entry that ONLY the carry path can
    produce -- the direct fetch_characters + from_characters path never asks
    for the intel or the related cast, so it can never emit that entry.

    Seeded-defect drill: rewire _GazetteerBuildThread.run back to
    `SeriesGazetteer.from_characters(AniListClient().fetch_characters(id))`
    -> the related entry vanishes from the cached map -> red.
    """
    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.gazetteer.gazetteer import SeriesGazetteer, gazetteer_path

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))  # empty -> cache miss
    monkeypatch.setattr(
        tray_mod, "AniListClient", lambda *a, **k: _IntelCarryClient()
    )

    tray, client = _tray(qapp)  # connected = True
    tray.on_series_chosen(_IntelCarryClient.PICKED, "Yani Neko")

    qtbot.waitUntil(
        lambda: any(c.get("action") == "set_series" for c in client.sent),
        timeout=3000,
    )
    gaz = SeriesGazetteer.load(gazetteer_path(_IntelCarryClient.PICKED))
    assert gaz is not None, "the pick wrote no gazetteer cache at all"
    assert gaz.lookup("ヤニねこ") == "Cigarette Cat", (
        "the picked media's own character is missing from the cached map"
    )
    assert gaz.lookup("ゼロねこ") == "Zero Cat", (
        "the PREQUEL cast never reached the cached gazetteer: the pick-time "
        "worker did not build through build_series_gazetteer, so the "
        "relations carry is shipped-but-never-launched"
    )
    qtbot.wait(100)  # drain the worker's finished->deleteLater


def test_on_series_chosen_persists_the_series_profile_beside_the_gazetteer(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """A cache-miss palette pick persists profile-<media_id>.json as a side
    effect of the build (build_profile runs inside build_series_gazetteer):
    load_profile must return the ingested profile afterwards. That file is
    the ONLY thing the translation.series_context_hint lever reads --
    without this write the flag is inert end-to-end.

    Seeded-defect drill: same revert as the carry test (direct
    fetch_characters + from_characters in run()) -> build_profile never
    runs -> profile-<id>.json is never written -> load_profile returns
    None -> red.
    """
    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.gazetteer.series_profile import load_profile

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))  # empty -> cache miss
    monkeypatch.setattr(
        tray_mod, "AniListClient", lambda *a, **k: _IntelCarryClient()
    )

    tray, client = _tray(qapp)  # connected = True
    tray.on_series_chosen(_IntelCarryClient.PICKED, "Yani Neko")

    qtbot.waitUntil(
        lambda: any(c.get("action") == "set_series" for c in client.sent),
        timeout=3000,
    )
    profile = load_profile(_IntelCarryClient.PICKED)
    assert profile is not None, (
        "the pick persisted no series profile: profile-<id>.json was never "
        "written, so translation.series_context_hint has nothing to read"
    )
    assert profile.synopsis_first_sentence == "A cat takes up smoking.", (
        f"persisted profile carries the wrong synopsis: "
        f"{profile.synopsis_first_sentence!r}"
    )
    assert [r.media_id for r in profile.relations] == [_IntelCarryClient.RELATED], (
        f"persisted profile carries the wrong relations: {profile.relations!r}"
    )
    qtbot.wait(100)  # drain the worker's finished->deleteLater


def test_on_series_chosen_rebuilds_a_pre_schema_v1_cache_through_the_full_build(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """A gazetteer cache written by the previously-shipped v1 pick path (a
    bare flat map: no schema, no built_at) reads as a MISS at pick time
    (cache_needs_rebuild): re-picking the series runs the FULL build again,
    so pre-versioning caches finally benefit from extractor fixes and the
    relations carry instead of being served frozen forever. The rewritten
    cache carries the current schema + a fresh built_at (the next pick
    inside 24 h is a plain hit), and the profile rides the same build
    (build_profile runs inside build_series_gazetteer).

    Seeded-defect drill: revert the worker's gate to
    ``if not path.is_file():`` -> the flat map short-circuits as a cache
    hit, the map keeps only the v1 entry and stays schema-less -> the
    carry-entry, replaced-entry, and needs-rebuild asserts all go red.
    """
    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.gazetteer.gazetteer import (
        SeriesGazetteer, cache_needs_rebuild, gazetteer_path,
    )
    from yamabiko.gazetteer.series_profile import load_profile

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    cache = gazetteer_path(_IntelCarryClient.PICKED)
    cache.parent.mkdir(parents=True, exist_ok=True)
    # The v1-era on-disk state: bare flat map, no profile beside it.
    cache.write_text('{"オールド": "Old Spelling"}', encoding="utf-8")

    monkeypatch.setattr(
        tray_mod, "AniListClient", lambda *a, **k: _IntelCarryClient()
    )

    tray, client = _tray(qapp)  # connected = True
    tray.on_series_chosen(_IntelCarryClient.PICKED, "Yani Neko")

    qtbot.waitUntil(
        lambda: any(c.get("action") == "set_series" for c in client.sent),
        timeout=3000,
    )
    gaz = SeriesGazetteer.load(cache)
    assert gaz is not None and gaz.lookup("ヤニねこ") == "Cigarette Cat", (
        "the v1 cache was not rebuilt through the full build"
    )
    assert gaz.lookup("ゼロねこ") == "Zero Cat", (
        "the rebuild skipped the relations carry"
    )
    assert gaz.lookup("オールド") is None, (
        "the rebuilt cache still serves the v1-era entry; the old map was "
        "kept instead of being replaced by the fresh build"
    )
    assert cache_needs_rebuild(cache) is False, (
        "the rebuilt cache was not stamped with the current schema/built_at"
    )
    assert load_profile(_IntelCarryClient.PICKED) is not None, (
        "the v1-cache rebuild did not persist the series profile"
    )
    qtbot.wait(100)  # drain the worker's finished->deleteLater


def test_profile_backfill_failure_degrades_to_the_plain_cache_hit(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """The missing-profile backfill beside a FRESH cache is best-effort: the
    single intel request must be ATTEMPTED, but when it fails (an offline
    user re-picking a series whose cache is healthy) the pick still succeeds
    as the plain cache hit it always was -- built fires and set_series is
    sent, and the still-missing profile retries on the next pick. Emitting
    buildFailed here would regress the cache hit's documented
    offline-friendliness. The backfill's throwaway client is closed even on
    the failure path (the worker's finally).

    Seeded-defect drill: remove the try/except around the backfill call ->
    the RuntimeError escapes to run()'s outer except -> buildFailed instead
    of built -> no set_series -> the waitUntil times out -> red. (The
    attempted-backfill assertion is red pre-fix on its own: the old run()
    never touches the client on a cache hit.) Drop the backfill's
    ``client.close()`` finally -> the closed assert goes red.
    """
    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.gazetteer.series_profile import load_profile

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    _write_fresh_cache(182205, {"リムル": "Rimuru"})
    # No profile beside it: backfill due, but the network is down.

    attempts: list[int] = []
    made: list = []

    class _OfflineClient:
        def __init__(self):
            self.closed = 0

        def fetch_media_intel(self, media_id):
            attempts.append(media_id)
            raise RuntimeError("network down")

        def fetch_characters(self, media_id, max_pages: int = 6, on_page=None):
            raise AssertionError("offline cache-hit pick refetched characters")

        def close(self):
            self.closed += 1

    monkeypatch.setattr(
        tray_mod, "AniListClient",
        lambda *a, **k: made.append(_OfflineClient()) or made[-1],
    )

    tray, client = _tray(qapp)  # connected = True
    tray.on_series_chosen(182205, "Slime (2024)")

    qtbot.waitUntil(
        lambda: any(c.get("action") == "set_series" for c in client.sent),
        timeout=3000,
    )
    assert attempts == [182205], (
        "no profile backfill was attempted over the profile-less cache"
    )
    assert load_profile(182205) is None, (
        "a failed backfill left a profile file behind; nothing valid could "
        "have been written from a dead intel request"
    )
    assert made and made[0].closed == 1, (
        "the failed backfill's AniList client was never closed"
    )
    qtbot.wait(100)  # drain the worker's finished->deleteLater


def test_re_pick_after_corrupt_profile_quarantine_backfills_a_fresh_profile(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """An externally-corrupted profile-<id>.json beside a healthy FRESH
    gazetteer cache heals across the real two-step path: the pipeline-side
    load_profile read QUARANTINES the corrupt file (returns None, moves it
    aside), and the next re-pick's cache-hit branch then sees the profile
    missing and backfills a fresh one with a single intel request. Both
    profile writers are existence-gated, so without the quarantine the
    corrupt file satisfies profile_path(id).is_file() on every re-pick and
    the series hint is dead until the cache's freshness horizon lapses.

    Seeded-defect drill: make load_profile return None WITHOUT moving the
    corrupt file aside (the pre-quarantine behavior) -> the re-pick skips
    the backfill branch (the file still 'exists'), no intel request is
    made, and the final load_profile is None again -> red.
    """
    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.gazetteer.series_profile import load_profile, profile_path

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    _write_fresh_cache(_IntelCarryClient.PICKED, {"ヤニねこ": "Cigarette Cat"})
    profile_path(_IntelCarryClient.PICKED).parent.mkdir(
        parents=True, exist_ok=True)
    profile_path(_IntelCarryClient.PICKED).write_text(
        "{corrupt beyond parsing", encoding="utf-8")

    # The pipeline-side read (translation_stage loads the profile once per
    # series epoch): corrupt -> None AND the file is moved aside so the
    # next pick can heal it.
    assert load_profile(_IntelCarryClient.PICKED) is None

    intel_calls: list[int] = []

    class _IntelOnlyClient(_IntelCarryClient):
        def fetch_media_intel(self, media_id: int) -> dict:
            intel_calls.append(media_id)
            return super().fetch_media_intel(media_id)

        def fetch_characters(self, media_id, max_pages: int = 6, on_page=None):
            raise AssertionError(
                "the corrupt-profile heal refetched characters despite a "
                "cached gazetteer; the backfill budget is ONE intel request"
            )

    monkeypatch.setattr(
        tray_mod, "AniListClient", lambda *a, **k: _IntelOnlyClient()
    )

    tray, client = _tray(qapp)  # connected = True
    tray.on_series_chosen(_IntelCarryClient.PICKED, "Yani Neko")

    qtbot.waitUntil(
        lambda: any(c.get("action") == "set_series" for c in client.sent),
        timeout=3000,
    )
    assert intel_calls == [_IntelCarryClient.PICKED], (
        "the re-pick after quarantine never made the single intel request "
        "that backfills the profile"
    )
    profile = load_profile(_IntelCarryClient.PICKED)
    assert profile is not None, (
        "the corrupt profile never healed: quarantine + re-pick backfill "
        "did not regenerate profile-<id>.json"
    )
    assert profile.synopsis_first_sentence == "A cat takes up smoking.", (
        f"healed profile carries the wrong synopsis: "
        f"{profile.synopsis_first_sentence!r}"
    )
    qtbot.wait(100)  # drain the worker's finished->deleteLater


def test_second_pick_while_build_in_flight_queues_the_latest_pick(qapp):
    """A second palette pick while a build is still running is a DIFFERENT
    request (unlike a diagnostics re-click, which means "same request"):
    it must not spawn a parallel worker, but it must not be silently
    dropped either -- the palette has already hidden itself, so a drop
    looks like the pick took and the in-flight build then activates the
    WRONG series. It queues instead, latest pick wins, and nothing is
    sent until a build for the queued pick completes.

    Seeded-defect drill: restore the bare early-return guard (drop the
    ``self._queued_pick = (media_id, label)`` assignment; the pre-fix
    code) -> _queued_pick stays None -> red. Drop the isRunning() guard
    entirely -> the pick replaces the in-flight thread -> the identity
    assertion goes red.
    """
    class _RunningThread:
        def isRunning(self) -> bool:
            return True

    tray, client = _tray(qapp)
    sentinel = _RunningThread()
    tray._gaz_thread = sentinel

    tray.on_series_chosen(1, "Some Series")

    assert tray._gaz_thread is sentinel, (
        "a mid-build pick replaced the in-flight build thread"
    )
    assert tray._queued_pick == (1, "Some Series", ""), (
        "a mid-build pick was dropped instead of queued"
    )

    tray.on_series_chosen(2, "Newer Series")  # the LATEST pick owns the queue
    assert tray._queued_pick == (2, "Newer Series", ""), (
        "a third pick did not supersede the queued second one"
    )
    assert all(c.get("action") != "set_series" for c in client.sent), (
        "a queued pick sent set_series before its build completed"
    )


def test_stale_finished_slot_never_deletes_a_replacement_thread(qapp):
    """A queued finished slot for build thread A must delete A itself, never
    whatever self._gaz_thread points at by the time the slot runs.

    QThread.finished is delivered to the GUI thread as a QUEUED call, AFTER the
    worker has flipped isRunning() to false. In that gap a re-pick passes
    on_series_chosen's isRunning() guard and installs a new RUNNING thread B in
    self._gaz_thread. If the stale slot deletes via the attribute it destroys B
    while it is still running (Qt6 qFatal 'Destroyed while thread is still
    running') and nulls the live ref out from under the next finish. The slot
    must delete the thread that actually emitted finished, and null the ref only
    when it still names that thread.

    Seeded-defect drill: revert the slot to `self._gaz_thread.deleteLater()`
    + unconditional `self._gaz_thread = None` -> new.deleted == 1 and the ref
    is nulled -> all three asserts go red (confirmed red against the unfixed
    slot).
    """
    tray, _ = _tray(qapp)

    class _Recorder:
        """Stand-in for a _GazetteerBuildThread: records deleteLater() so the
        test observes WHICH thread the stale slot schedules for destruction."""

        def __init__(self):
            self.deleted = 0

        def deleteLater(self):
            self.deleted += 1

    old, new = _Recorder(), _Recorder()
    # Interleaving under test: thread A finished (its slot is queued), the user
    # re-picked so _gaz_thread already points at the running thread B, THEN A's
    # queued slot runs -- for the OLD thread it was bound to.
    tray._gaz_thread = new
    tray._on_gaz_thread_finished(old)

    assert old.deleted == 1, "the thread that actually finished was not deleted"
    assert new.deleted == 0, (
        "the stale finished slot deleted the running replacement thread "
        "(Qt6 qFatal: Destroyed while thread is still running)"
    )
    assert tray._gaz_thread is new, (
        "the stale finished slot nulled the live replacement thread's ref"
    )


def test_on_series_chosen_second_pick_in_same_session_also_sends_set_series(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """Two sequential palette picks in ONE overlay session must BOTH switch
    the series -- users change series across episodes without relaunching.

    The retained _GazetteerBuildThread wrapper schedules its own deletion on
    finish; the finished slot must ALSO null self._gaz_thread. Otherwise the
    re-entrancy guard on the SECOND pick
    (`self._gaz_thread is not None and self._gaz_thread.isRunning()`) calls
    isRunning() on the already-deleted C++ wrapper and raises
    `RuntimeError: Internal C++ object (_GazetteerBuildThread) already
    deleted` before the second build ever starts -- the use-after-delete that
    limited series changes to exactly one per launch.

    Both gazetteers (and their profiles) are pre-cached on disk and
    AniListClient is patched to explode, so each worker takes the offline
    cache-hit branch and no network is touched on either pick (proving the
    second build really ran, not that a stale thread was reused).

    Seeded-defect drill: revert the finished wiring to
    `thread.finished.connect(thread.deleteLater)` (dropping the ref reset in
    _on_gaz_thread_finished) -> after pick A finishes, _gaz_thread stays a
    zombie wrapper that never becomes None, so `waitUntil(_gaz_thread is
    None)` after pick A times out -> red (and were the loop to reach pick B,
    its guard would call isRunning() on the deleted wrapper and raise
    RuntimeError). Confirmed red against the unfixed wiring: waitUntil timed
    out after pick A.
    """
    from yamabiko.overlay_app import tray as tray_mod

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    for media_id, hint in (
        (182205, {"リムル": "Rimuru"}),
        (170695, {"アニス": "Anis"}),
    ):
        _write_fresh_cache(media_id, hint)
        _write_profile_stub(media_id)  # both present -> zero requests

    def _no_network(*a, **k):
        raise AssertionError("worker hit AniList despite a cached gazetteer")

    monkeypatch.setattr(tray_mod, "AniListClient", _no_network)

    tray, client = _tray(qapp)  # connected = True

    # --- Pick A: first series change of the session ---
    tray.on_series_chosen(182205, "Slime (2024)")
    # Spin the loop until the build finished AND the finished slot reset the
    # retained-wrapper ref -- the reset is exactly what keeps pick B's guard
    # from reading a deleted C++ object.
    qtbot.waitUntil(lambda: tray._gaz_thread is None, timeout=3000)
    sent_a = [c for c in client.sent if c.get("action") == "set_series"]
    assert sent_a[-1:] == [{
        "type": "command", "action": "set_series",
        "media_id": 182205, "label": "Slime (2024)", "native": None,
    }], f"first pick did not send set_series; got {client.sent}"

    # --- Pick B: second series change (the path with ZERO prior coverage;
    # the guard here used to raise RuntimeError on the deleted wrapper) ---
    tray.on_series_chosen(170695, "Tearmoon (2023)")
    qtbot.waitUntil(lambda: tray._gaz_thread is None, timeout=3000)
    sent_b = [c for c in client.sent if c.get("action") == "set_series"]
    assert len(sent_b) == 2 and sent_b[-1] == {
        "type": "command", "action": "set_series",
        "media_id": 170695, "label": "Tearmoon (2023)", "native": None,
    }, f"second pick did not send a second set_series; got {client.sent}"


# --- Queued pick: a second pick during an in-flight build must win (P1-7) ---


class _GatedBuildClient:
    """AniListClient stand-in whose FIRST wire call (the intel request)
    blocks on a gate, holding the build worker in the exact window where a
    second palette pick can land. ``fail_after_gate`` makes the released
    build fail (the queued-pick failure leg)."""

    def __init__(self, entered, gate, fail_after_gate=False):
        self._entered = entered
        self._gate = gate
        self._fail = fail_after_gate
        self.closed = 0

    def fetch_media_intel(self, media_id):
        self._entered.set()
        self._gate.wait(10)  # self-releasing: a failed assert can't wedge it
        if self._fail:
            raise RuntimeError("network down")
        return {"genres": [], "tags": [], "description": "", "relations": []}

    def fetch_characters(self, media_id, max_pages: int = 6, on_page=None):
        from yamabiko.gazetteer.anilist import Character
        return [Character("ヤニ", "Yani", [], role="MAIN")]

    def close(self):
        self.closed += 1


def test_second_pick_during_inflight_build_activates_only_the_latest_pick(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """The user picks series A, then picks series B while A's build is still
    fetching: B must win. A's completed build is DISCARDED -- no set_series,
    no label flip, no auto-genre ride-along for A (the palette already hid
    itself after pick B, so activating A would silently flip the user to the
    WRONG series) -- and a build for B starts as soon as A's finishes,
    sending set_series for B alone.

    Drives REAL workers: A's build blocks in the fake AniList client on a
    gate; B is a fresh cache+profile hit needing no network.

    Seeded-defect drill: restore on_series_chosen's early-return drop (the
    pre-fix guard) -> pick B vanishes, A's build completes and sends
    set_series for A -> the picks assertion goes red (this IS the pre-fix
    code; confirmed red against it).
    """
    import threading

    from yamabiko.overlay_app import tray as tray_mod

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    entered, gate = threading.Event(), threading.Event()
    monkeypatch.setattr(
        tray_mod, "AniListClient",
        lambda *a, **k: _GatedBuildClient(entered, gate),
    )
    _write_fresh_cache(170695, {"アニス": "Anis"})  # pick B: plain hit
    _write_profile_stub(170695)

    tray, client = _tray(qapp, on_change_series=lambda: None)
    try:
        tray.on_series_chosen(182205, "Slime (2024)")    # pick A: blocked
        qtbot.waitUntil(entered.is_set, timeout=3000)
        tray.on_series_chosen(170695, "Tearmoon (2023)")  # pick B mid-build

        gate.set()  # A's build completes AFTER pick B
        qtbot.waitUntil(
            lambda: any(c.get("action") == "set_series" for c in client.sent),
            timeout=5000,
        )
        qtbot.waitUntil(lambda: tray._gaz_thread is None, timeout=5000)
        qtbot.wait(150)  # drain: a stray set_series for A would land here

        picks = [c for c in client.sent if c.get("action") == "set_series"]
        assert picks == [{
            "type": "command", "action": "set_series",
            "media_id": 170695, "label": "Tearmoon (2023)", "native": None,
        }], (
            f"a pick made during an in-flight build did not supersede the "
            f"finished build's activation; set_series commands: {picks}"
        )
        assert tray._flyout._series_text.title() == "Tearmoon", (
            "the stale build flipped the series row"
        )
    finally:
        gate.set()


def test_failed_inflight_build_starts_the_queued_pick_without_a_stale_toast(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """Pick A's build FAILS after the user already re-picked B: the stale
    failure must not toast 'Series not changed' for A (the user's current
    intent is B, and B succeeds), and B's queued build starts immediately,
    sending set_series for B.

    Seeded-defect drill: drop the queued-pick check from
    _on_gazetteer_build_failed -> a toast for A fires -> the no-toast
    assertion goes red. Restore the pre-fix early-return drop in
    on_series_chosen -> B vanishes, no set_series is ever sent -> the
    waitUntil times out -> red.
    """
    import threading

    from yamabiko.overlay_app import tray as tray_mod

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    entered, gate = threading.Event(), threading.Event()
    # Pick A has NO cache, so its failure has no fallback: pre-queueing it
    # would have been the toast + series-unchanged outcome.
    monkeypatch.setattr(
        tray_mod, "AniListClient",
        lambda *a, **k: _GatedBuildClient(entered, gate, fail_after_gate=True),
    )
    _write_fresh_cache(170695, {"アニス": "Anis"})  # pick B: plain hit
    _write_profile_stub(170695)

    tray, client = _tray(qapp)
    shown: list[tuple] = []
    monkeypatch.setattr(tray._tray, "showMessage", lambda *a, **k: shown.append(a))
    try:
        tray.on_series_chosen(99999, "Doomed Series")     # pick A: will fail
        qtbot.waitUntil(entered.is_set, timeout=3000)
        tray.on_series_chosen(170695, "Tearmoon (2023)")  # pick B mid-build

        gate.set()  # A's build now fails, with B queued
        qtbot.waitUntil(
            lambda: any(c.get("action") == "set_series" for c in client.sent),
            timeout=5000,
        )
        qtbot.waitUntil(lambda: tray._gaz_thread is None, timeout=5000)
        qtbot.wait(150)  # drain: a stray failure toast would land here

        assert shown == [], (
            f"a superseded build's failure still raised a toast for the "
            f"abandoned pick: {shown}"
        )
        picks = [c for c in client.sent if c.get("action") == "set_series"]
        assert picks == [{
            "type": "command", "action": "set_series",
            "media_id": 170695, "label": "Tearmoon (2023)", "native": None,
        }], f"the queued pick did not activate after the failed build: {picks}"
    finally:
        gate.set()


def test_clear_during_inflight_build_is_not_overridden_when_the_build_finishes(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """Decision 11: a Clear while pick A's build is still fetching, with pick
    B queued behind it, is the user's latest intent: A's finished build must
    not re-activate A and the queued B must not start. Only the clear's
    set_series goes out and the row keeps "No series".

    Seeded defects: keep _queued_pick on Clear (B builds and sends its
    set_series) or let the finished build activate (set_series and the
    auto-genre for A) -> red."""
    import threading

    from yamabiko.overlay_app import tray as tray_mod

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    entered, gate = threading.Event(), threading.Event()
    monkeypatch.setattr(
        tray_mod, "AniListClient",
        lambda *a, **k: _GatedBuildClient(entered, gate),
    )
    _write_fresh_cache(170695, {"アニス": "Anis"})  # pick B: plain hit
    _write_profile_stub(170695)

    tray, client = _tray(qapp, on_change_series=lambda: None)
    try:
        tray.on_series_chosen(182205, "Slime (2024)")    # pick A: blocked
        qtbot.waitUntil(entered.is_set, timeout=3000)
        tray.on_series_chosen(170695, "Tearmoon (2023)")  # pick B: queued
        tray._flyout.clearSeriesRequested.emit()

        gate.set()  # A's build completes AFTER the Clear
        qtbot.waitUntil(lambda: tray._gaz_thread is None, timeout=5000)
        qtbot.wait(150)  # drain: a stray activation or queued build lands here

        assert [c for c in client.sent if c.get("action") in ("set_series", "set_genre")] == [
            {"type": "command", "action": "set_series", "media_id": None},
        ], f"the finished build overrode the Clear: {client.sent}"
        assert tray._flyout._series_text.title() == "No series"
    finally:
        gate.set()


# --- the series bar: the tray drives its status, Clear and reduce motion ---


def _apothecary_hit():
    from yamabiko.gazetteer.anilist import SeriesHit

    return SeriesHit(161645, "Kusuriya no Hitorigoto", "The Apothecary Diaries",
                     "薬屋のひとりごと", 2023, "TV", cover_image=None, color="#e4a128")


def _series_bar(qtbot, hit, *, reduce_motion: bool = True):
    """A real SeriesPalette (the series bar) whose AniList client answers
    every search with ``hit`` and reads no watch list (no username): no
    network. Reduced motion, so every view and status change lands at once."""
    from yamabiko.overlay_app.series_palette import SeriesPalette

    class _Client:
        def search_series(self, query, limit=8):
            return [hit]

        def fetch_watching(self, username):
            return []

    bar = SeriesPalette(client=_Client(), username_provider=lambda: "",
                        reduce_motion=reduce_motion)
    qtbot.addWidget(bar)
    return bar


def _pick_in_bar(qtbot, bar, hit, query: str = "kusuriya") -> None:
    """The user's pick: summon the bar, settle a search answer for ``query``
    (the field text set with its signals blocked, so the new text arms no
    debounced search; a re-summon does re-run the search for the field's
    old text at once, against the bar's stub client, and that answer is
    dropped as stale), and press Enter on the lit first row. Leaves the bar
    in its status face for ``hit`` (and emits series_chosen)."""
    from PySide6.QtCore import Qt

    bar.summon()
    bar._search.blockSignals(True)
    bar._search.setText(query)
    bar._search.blockSignals(False)
    bar._on_results(query, True, [hit])
    qtbot.keyClick(bar._search, Qt.Key_Return)
    assert bar._mode == "status" and bar._status_media_id == hit.media_id, (
        "Enter on the lit result did not leave the bar in the pick's status"
    )


class _PagedBuildClient:
    """AniListClient stand-in for one pick's cache-miss build: the intel
    request answers at once, the cast's first page reports half the cast,
    then the build waits on ``gate`` before its last page."""

    def __init__(self, gate):
        self._gate = gate

    def fetch_media_intel(self, media_id):
        return {"genres": [], "tags": [], "description": "", "relations": []}

    def fetch_characters(self, media_id, max_pages: int = 6, on_page=None):
        from yamabiko.gazetteer.anilist import Character

        on_page(0.5)
        self._gate.wait(10)  # self-releasing: a failed assert can't wedge it
        on_page(1.0)
        return [Character("猫猫", "Maomao", [], role="MAIN")]

    def close(self):
        pass


def test_pick_build_progress_and_success_drive_the_series_bar_status(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """L9/L10: after a pick in the bar, the tray's build feeds the bar's
    status: the build's staged progress moves the loading line (0.45 once
    half the cast is in), and the finished build turns the status into
    "Now watching <short title>." with the line full.

    Seeded defects: never connect the build's progress signal (or never
    forward it) -> the line stays at 0 -> the first waitUntil times out;
    drop show_built from _on_gazetteer_built -> the status never finishes
    -> red.
    """
    import threading

    from yamabiko.overlay_app import tray as tray_mod

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))  # empty: a cache-miss build
    gate = threading.Event()
    monkeypatch.setattr(tray_mod, "AniListClient", lambda *a, **k: _PagedBuildClient(gate))
    hit = _apothecary_hit()
    bar = _series_bar(qtbot, hit)
    tray, client = _tray(qapp, on_change_series=bar.summon, series_palette=bar)
    bar.series_chosen.connect(tray.on_series_chosen)  # as build_overlay wires it
    try:
        _pick_in_bar(qtbot, bar, hit)
        assert bar._status.text() == "Loading character names for Apothecary Diaries\u2026"
        qtbot.waitUntil(lambda: bar._fraction_target == pytest.approx(0.45), timeout=3000)

        gate.set()
        qtbot.waitUntil(bar._status.is_done, timeout=5000)
        assert bar._status.text() == "Now watching Apothecary Diaries."
        assert bar._fraction_target == pytest.approx(1.0)
        assert any(c.get("action") == "set_series" and c.get("media_id") == 161645
                   for c in client.sent)
    finally:
        gate.set()
        qtbot.waitUntil(lambda: tray._gaz_thread is None, timeout=5000)


def test_failed_pick_build_hides_the_series_bar_and_toasts_series_not_changed(
    qapp, qtbot, monkeypatch,
):
    """L9: when the pick's build fails, the bar's loading status goes at once
    and the tray's "Series not changed" notification is the notice.

    Seeded defect: drop show_build_failed from _on_gazetteer_build_failed ->
    the bar keeps showing "Loading character names for …" -> red.
    """
    from yamabiko.gazetteer.anilist import AniListUnavailable

    hit = _apothecary_hit()
    bar = _series_bar(qtbot, hit)
    tray, _client = _tray(qapp, on_change_series=bar.summon, series_palette=bar)
    shown: list[tuple] = []
    monkeypatch.setattr(tray._tray, "showMessage", lambda *a, **k: shown.append(a))
    _pick_in_bar(qtbot, bar, hit)

    tray._on_gazetteer_build_failed(
        hit.media_id, "The Apothecary Diaries (2023)", AniListUnavailable("down"),
    )
    assert not bar.isVisible(), "the bar kept its loading status after the build failed"
    assert [a[0] for a in shown] == ["Series not changed"]


def test_series_bar_clear_sends_set_series_without_a_series_and_keeps_the_bar_open(
    qapp, qtbot,
):
    """L8: the chip's Clear is the flyout's Clear: set_series with a null
    media id goes to the pipeline, the chip hides, and the bar stays open.

    Seeded defects: never connect clearRequested in Tray.__init__ -> nothing
    is sent -> red; make cancel_status hide the bar in its search face ->
    the bar closes -> red.
    """
    from PySide6.QtWidgets import QAbstractButton

    hit = _apothecary_hit()
    bar = _series_bar(qtbot, hit)
    tray, client = _tray(qapp, on_change_series=bar.summon, series_palette=bar)
    tray._on_series_received(hit.media_id, "The Apothecary Diaries (2023)", "薬屋のひとりごと")
    bar.summon()
    assert not bar._chip.isHidden(), "precondition: the chip shows the active series"

    clear = next(b for b in bar._chip.findChildren(QAbstractButton) if b.text() == "Clear")
    clear.click()
    assert [c for c in client.sent if c.get("action") == "set_series"] == [
        {"type": "command", "action": "set_series", "media_id": None},
    ]
    assert bar._chip.isHidden(), "the chip still shows a series after Clear"
    assert bar.isVisible(), "Clear closed the bar"


def test_a_superseded_build_moves_no_series_bar_status(qapp, qtbot, tmp_path, monkeypatch):
    """L9 (Decision 70): the bar's status follows the newest pick. Picking
    the same series again while its first build still runs queues the
    re-pick and supersedes that build: the first build's progress must not
    move the new status line, and the re-pick's own build then finishes
    the status.

    Seeded defect: forward progress without the supersede check -> the
    stale 0.6 lands on the fresh status line -> red.
    """
    import threading

    from yamabiko.overlay_app import tray as tray_mod

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    entered, gate = threading.Event(), threading.Event()
    monkeypatch.setattr(
        tray_mod, "AniListClient", lambda *a, **k: _GatedBuildClient(entered, gate),
    )
    hit = _apothecary_hit()
    bar = _series_bar(qtbot, hit)
    tray, _client = _tray(qapp, on_change_series=bar.summon, series_palette=bar)
    bar.series_chosen.connect(tray.on_series_chosen)
    try:
        _pick_in_bar(qtbot, bar, hit)                       # build 1 blocks in its intel request
        qtbot.waitUntil(entered.is_set, timeout=3000)
        _pick_in_bar(qtbot, bar, hit, query="kusuriya no hitorigoto")  # the re-pick: queued
        assert bar._fraction_target == 0.0                  # precondition: a fresh status line

        tray._on_gazetteer_progress(hit.media_id, 0.6)      # build 1's report, as its signal delivers it
        assert bar._fraction_target == 0.0, "a superseded build moved the new pick's loading line"

        gate.set()  # build 1 ends (discarded); the queued re-pick builds from its fresh cache
        qtbot.waitUntil(bar._status.is_done, timeout=5000)
    finally:
        gate.set()
        qtbot.waitUntil(lambda: tray._gaz_thread is None, timeout=5000)


def test_restored_series_shows_in_the_series_bar_chip_with_its_saved_cover(qapp, qtbot):
    """L8/L11: the series the pipeline restores at boot (its state frame on
    connect) is the bar's current series: the chip shows its short title,
    the full title as its tooltip and, offline, the poster an earlier
    session saved for it instead of the neutral tile.

    Seeded defects: _set_series stops telling the palette -> the chip stays
    hidden -> red; cover_for skips the saved file -> the chip shows the
    neutral tile, no poster pixels -> red.
    """
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QColor, QImage, QPixmap, QRegion
    from PySide6.QtWidgets import QWidget

    from yamabiko.gazetteer.gazetteer import cover_path
    from yamabiko.overlay_app.series_palette import save_cover_file
    from yamabiko.overlay_app.tray import Tray

    poster = QPixmap(10, 14)
    poster.fill(QColor("#123456"))
    assert save_cover_file(poster, cover_path(161645))  # what an earlier session saved
    bar = _series_bar(qtbot, _apothecary_hit())
    client = _make_signal_client_class()()
    tray = Tray(client, on_quit=lambda: None, on_change_series=bar.summon,
                series_palette=bar)
    client.seriesReceived.emit(161645, "The Apothecary Diaries (2023)", "薬屋のひとりごと")
    bar.summon()

    chip = bar._chip
    assert not chip.isHidden(), f"the restored series is not in the bar's chip (tray {tray!r})"
    assert chip.title() == "Apothecary Diaries"
    assert chip.toolTip() == "The Apothecary Diaries"
    image = QImage(chip.size(), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    chip.render(image, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    poster_px = sum(
        image.pixelColor(x, y) == QColor("#123456")
        for x in range(image.width()) for y in range(image.height())
    )
    assert poster_px >= 20 * 28 // 2, (
        f"the chip shows no saved poster ({poster_px} poster-colored pixels)"
    )


def test_manual_genre_after_queued_pick_is_not_clobbered_by_its_ride_along(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """The manual-pick recency snapshot keys on PICK time for QUEUED picks
    too: a manual genre chosen AFTER pick B was queued (while A's build was
    still running) is chronologically newer than B, so B's eventual built
    must skip the auto-genre ride-along -- set_series for B still goes out,
    but the only genre command is the manual one.

    Seeded-defect drill: move the ``self._series_pick_genre_seq =
    self._manual_genre_seq`` snapshot out of on_series_chosen and into
    _start_gazetteer_build (snapshot at build START instead of pick time)
    -> B's build starts after the manual pick and re-snapshots the counter
    -> the ride-along sends 'anime' over the fresher manual 'drama' -> the
    exactly-one-genre-command assertion goes red.
    """
    import threading

    from yamabiko.overlay_app import tray as tray_mod

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    entered, gate = threading.Event(), threading.Event()
    monkeypatch.setattr(
        tray_mod, "AniListClient",
        lambda *a, **k: _GatedBuildClient(entered, gate),
    )
    _write_fresh_cache(170695, {"アニス": "Anis"})  # pick B: plain hit
    _write_profile_stub(170695)

    tray, client = _tray(qapp)  # auto_genre defaults True; connected = True
    try:
        tray.on_series_chosen(182205, "Slime (2024)")     # pick A: blocked
        qtbot.waitUntil(entered.is_set, timeout=3000)
        tray.on_series_chosen(170695, "Tearmoon (2023)")  # pick B queued
        tray._on_genre_chosen("drama")  # manual pick, AFTER pick B

        gate.set()
        qtbot.waitUntil(
            lambda: any(c.get("action") == "set_series" for c in client.sent),
            timeout=5000,
        )
        qtbot.waitUntil(lambda: tray._gaz_thread is None, timeout=5000)
        qtbot.wait(150)  # drain: a stray set_genre anime would land here

        genres = [c for c in client.sent if c.get("action") == "set_genre"]
        assert genres == [
            {"type": "command", "action": "set_genre", "genre": "drama"}
        ], (
            "the queued pick's auto-genre ride-along clobbered a manual "
            f"genre pick made during its wait; genre commands: {genres}"
        )
        picks = [c for c in client.sent if c.get("action") == "set_series"]
        assert [c["media_id"] for c in picks] == [170695]
    finally:
        gate.set()


# --- Cache freshness at pick time (P1-9) ---


def test_pick_with_cache_older_than_24h_rebuilds_and_restamps_freshness(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """A current-schema cache OLDER than the 24 h freshness horizon is
    rebuilt on the next pick -- an airing series' cast must not stay frozen
    at first-pick state forever -- and the rewrite restamps built_at, so
    the pick after that is a plain hit again.

    Seeded-defect drill: revert the worker's gate to ``path.is_file()`` (or
    drop the built_at age check from cache_needs_rebuild) -> the stale
    envelope short-circuits, the new cast never lands -> red.
    """
    import time as _time

    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.gazetteer.gazetteer import (
        SeriesGazetteer, cache_needs_rebuild, gazetteer_path,
    )

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    cache = gazetteer_path(_IntelCarryClient.PICKED)
    cache.parent.mkdir(parents=True, exist_ok=True)
    SeriesGazetteer({"オールド": "Old Cast"}).save(
        cache, now=lambda: _time.time() - 25 * 60 * 60)
    # Profile present: ONLY staleness triggers the rebuild here.
    _write_profile_stub(_IntelCarryClient.PICKED)

    monkeypatch.setattr(
        tray_mod, "AniListClient", lambda *a, **k: _IntelCarryClient()
    )
    tray, client = _tray(qapp)  # connected = True
    tray.on_series_chosen(_IntelCarryClient.PICKED, "Yani Neko")

    qtbot.waitUntil(
        lambda: any(c.get("action") == "set_series" for c in client.sent),
        timeout=3000,
    )
    gaz = SeriesGazetteer.load(cache)
    assert gaz is not None and gaz.lookup("ヤニねこ") == "Cigarette Cat", (
        "a 25 h old cache was served as-is; an airing series' cast stays "
        "frozen at first-pick state"
    )
    assert cache_needs_rebuild(cache) is False, (
        "the rebuild did not restamp built_at; every future pick would "
        "refetch"
    )
    qtbot.wait(100)  # drain the worker's finished->deleteLater


def test_pick_rebuild_failure_falls_back_to_the_stale_cache(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """Offline re-pick of a series with a STALE cache: the rebuild attempt
    fails and the pick degrades to the old cache -- built fires, set_series
    goes out, NO failure toast, and the cache file survives byte-for-byte
    for the pipeline to load. A stale cache is strictly better than the
    cold-offline outcome (toast + series unchanged), which stays reserved
    for the no-cache-at-all case. The failed rebuild's client is still
    closed.

    Seeded-defect drill: drop the fallback try/except around the rebuild
    (let the failure escape to buildFailed; the pre-freshness shape) -> a
    toast fires and no set_series is ever sent -> the waitUntil times out
    -> red.
    """
    import time as _time

    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.gazetteer.gazetteer import SeriesGazetteer, gazetteer_path

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    cache = gazetteer_path(182205)
    cache.parent.mkdir(parents=True, exist_ok=True)
    SeriesGazetteer({"リムル": "Rimuru"}).save(
        cache, now=lambda: _time.time() - 25 * 60 * 60)
    before = cache.read_text(encoding="utf-8")

    made: list = []

    class _DownClient:
        def __init__(self):
            self.closed = 0

        def fetch_media_intel(self, media_id):
            raise RuntimeError("network down")

        def fetch_characters(self, media_id, max_pages: int = 6, on_page=None):
            raise RuntimeError("network down")

        def close(self):
            self.closed += 1

    monkeypatch.setattr(
        tray_mod, "AniListClient",
        lambda *a, **k: made.append(_DownClient()) or made[-1],
    )
    tray, client = _tray(qapp)  # connected = True
    shown: list[tuple] = []
    monkeypatch.setattr(tray._tray, "showMessage", lambda *a, **k: shown.append(a))

    tray.on_series_chosen(182205, "Slime (2024)")
    qtbot.waitUntil(
        lambda: any(c.get("action") == "set_series" for c in client.sent),
        timeout=3000,
    )
    qtbot.wait(100)  # drain: a stray failure toast would land here
    assert shown == [], (
        f"a rebuild failure over a usable stale cache raised a failure "
        f"toast: {shown}"
    )
    assert cache.read_text(encoding="utf-8") == before, (
        "the failed rebuild modified the stale cache; the fallback must "
        "leave the old file untouched"
    )
    assert made and made[0].closed == 1, (
        "the failed rebuild's AniList client was never closed"
    )


def test_rebuild_failure_over_corrupt_cache_notifies_instead_of_silent_switch(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """Re-pick of a series whose on-disk cache is CORRUPT while the rebuild
    fails: there is no usable fallback (the corrupt file loads to None,
    i.e. zero hints), so the pick must surface the 'Series not changed'
    toast and send NO set_series -- the no-silent-hintless contract the
    buildFailed path exists for. The corrupt file is quarantined aside so
    the next pick refetches instead of tripping over it again.

    Seeded defect: gate the fallback on ``path.is_file()`` instead of the
    cache actually loading (the pre-fix shape) -> built fires over the
    corrupt file, set_series goes out, no toast -> the toast wait times
    out -> red.
    """
    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.gazetteer.gazetteer import gazetteer_path

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    cache = gazetteer_path(182205)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("{corrupt beyond parsing", encoding="utf-8")

    class _DownClient:
        def fetch_media_intel(self, media_id):
            raise RuntimeError("network down")

        def fetch_characters(self, media_id, max_pages: int = 6, on_page=None):
            raise RuntimeError("network down")

        def close(self):
            pass

    monkeypatch.setattr(
        tray_mod, "AniListClient", lambda *a, **k: _DownClient(),
    )
    tray, client = _tray(qapp)  # connected = True
    shown: list[tuple] = []
    monkeypatch.setattr(tray._tray, "showMessage", lambda *a, **k: shown.append(a))

    tray.on_series_chosen(182205, "Slime (2024)")
    qtbot.waitUntil(lambda: bool(shown), timeout=3000)
    assert "not changed" in shown[0][0].lower(), (
        f"expected a 'Series not changed' notification; got {shown[0]}"
    )
    qtbot.wait(100)  # drain: a stray built->set_series would land here
    assert all(c.get("action") != "set_series" for c in client.sent), (
        "a failed rebuild over a CORRUPT cache silently switched the "
        f"series to a hint-less state: {client.sent}"
    )
    assert not cache.is_file(), (
        "the corrupt cache was left in place; the next pick would trip "
        "over it again instead of refetching"
    )
    qtbot.wait(50)  # finished->deleteLater drain


# --- Per-pick client hygiene + failure-copy wording ---


def test_build_worker_closes_its_anilist_client_after_a_successful_build(
    qapp, qtbot, tmp_path, monkeypatch,
):
    """The per-pick AniList client is closed in the worker's finally once
    the build finishes: one httpx.Client (and its sockets) per pick,
    released deterministically instead of leaking until GC.

    Seeded-defect drill: drop the ``client.close()`` finally from the
    rebuild branch of _GazetteerBuildThread.run -> closed stays 0 -> red.
    """
    from yamabiko.overlay_app import tray as tray_mod

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))  # empty -> rebuild
    made: list[_IntelCarryClient] = []
    monkeypatch.setattr(
        tray_mod, "AniListClient",
        lambda *a, **k: made.append(_IntelCarryClient()) or made[-1],
    )
    tray, client = _tray(qapp)  # connected = True
    tray.on_series_chosen(_IntelCarryClient.PICKED, "Yani Neko")
    qtbot.waitUntil(
        lambda: any(c.get("action") == "set_series" for c in client.sent),
        timeout=3000,
    )
    assert made and made[0].closed == 1, (
        "the successful build's AniList client was never closed"
    )
    qtbot.wait(100)  # drain the worker's finished->deleteLater


@pytest.mark.parametrize(
    "kind, expect_connection_copy",
    [
        pytest.param("unavailable", True, id="anilist-unavailable"),
        pytest.param("transport", True, id="httpx-transport"),
        pytest.param("other", False, id="non-network"),
    ],
)
def test_build_failure_toast_differentiates_network_from_other_failures(
    qapp, monkeypatch, kind, expect_connection_copy,
):
    """Network-class failures (AniListUnavailable, httpx transport errors)
    keep the 'Check your connection' copy; anything else (disk full, parse
    bug) names the exception class instead, so a bug report's triage isn't
    misdirected at the user's router.

    Seeded-defect drill: revert _on_gazetteer_build_failed to the fixed
    connection string -> the non-network row still says 'connection' and
    never names the class -> red.
    """
    import httpx as _httpx

    from yamabiko.gazetteer.anilist import AniListUnavailable

    exc = {
        "unavailable": AniListUnavailable("outage"),
        "transport": _httpx.ConnectError("no route to graphql.anilist.co"),
        "other": ValueError("bad payload"),
    }[kind]
    tray, _client = _tray(qapp)
    shown: list[tuple] = []
    monkeypatch.setattr(tray._tray, "showMessage", lambda *a, **k: shown.append(a))

    tray._on_gazetteer_build_failed(4321, "Yani Neko", exc)

    assert shown and shown[0][0] == "Series not changed"
    body = shown[0][1]
    if expect_connection_copy:
        assert "Check your connection" in body, (
            f"network-class failure lost the connection copy: {body!r}"
        )
        assert type(exc).__name__ not in body
    else:
        assert "Couldn't build the series data: ValueError" in body, (
            f"non-network failure does not name the exception class: {body!r}"
        )
        assert "connection" not in body.lower(), (
            f"non-network failure still blames the connection: {body!r}"
        )
