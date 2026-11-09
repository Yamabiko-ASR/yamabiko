"""Audio page tests. Device enumeration is abstracted via a callable
passed into the page so tests don't touch real audio hardware.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")


def _fake_devices() -> list[dict]:
    return [
        {"index": 0, "name": "Speakers (Realtek) [Loopback]"},
        {"index": 1, "name": "Headphones (USB) [Loopback]"},
    ]


def _row_names(page) -> list[str]:
    """The device rows' UIA names, top to bottom."""
    return [page.device_list.segment(dev["name"]).accessibleName() for dev in page._devices]


@pytest.fixture(autouse=True)
def hermetic_merged_config(monkeypatch):
    """Pin the merged-config read to the shipped default for every test in
    this module. AudioPage.initializePage now consults audio.device on BOTH
    legs (device seeding + the zero-device fallback), so an unpinned read
    would let the host machine's real user_overrides.yaml flip selection
    assertions whenever its saved device happens to match a fake name.
    Tests exercising the saved-device legs re-patch with their own values.
    """
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"audio": {"device": "default"}},
    )


def test_page_lists_devices_without_the_loopback_suffix_and_preselects_the_system_default(qtbot):
    """L8: each row shows the output's name without " [Loopback]", the
    Windows default carries "System default" (UIA name "<name>, System
    default"), and a first run with no saved device selects it; the value
    accept() persists stays the raw PyAudio name.

    Seeded defects: keep the suffix in the label, or preselect the first
    row instead of the system default -> red.
    """
    from yamabiko.wizard_pages.audio import AudioPage

    devices = [
        {"index": 0, "name": "Speakers (Realtek) [Loopback]"},
        {"index": 1, "name": "Headphones (USB) [Loopback]", "default": True},
    ]
    page = AudioPage(list_devices=lambda: devices)
    qtbot.addWidget(page)
    page.initializePage()

    assert _row_names(page) == ["Speakers (Realtek)", "Headphones (USB), System default"]
    assert page.selected_device_name() == "Headphones (USB) [Loopback]"
    assert page.test_button.isEnabled()


def test_page_incomplete_when_no_devices(qtbot, monkeypatch):
    # Hermetic: the zero-device leg now consults the merged config for a
    # saved device; pin the shipped default so a developer machine's real
    # user_overrides.yaml can't flip this first-run assertion.
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"audio": {"device": "default"}},
    )
    from yamabiko.wizard_pages.audio import AudioPage

    page = AudioPage(list_devices=lambda: [])
    qtbot.addWidget(page)
    page.initializePage()
    assert not page.isComplete()
    assert page.no_device_alert.message() == "No speakers or headphones found."
    assert page.no_device_alert.help_text() == "Connect one, then click Look again."
    assert not page.test_button.isEnabled()


def test_initialize_page_survives_device_enum_error(qtbot, monkeypatch):
    """If device enumeration raises (e.g. the Windows Audio service is stopped
    -> PyAudio()/PortAudio init raises OSError), initializePage must NOT
    propagate -- that exception escapes wiz.exec() and crashes the whole
    first-run wizard. It should fall into the existing no-device guidance.

    Seeded-defect drill: remove the try/except around self._list_devices() in
    initializePage -> the OSError propagates -> this test errors.
    """
    # Hermetic: pin the merged config at the shipped default so a saved
    # device on the host machine can't legitimately complete the page
    # (that leg has its own tests below).
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"audio": {"device": "default"}},
    )
    from yamabiko.wizard_pages.audio import AudioPage

    def _raising():
        raise OSError("Windows Audio service is not running")

    page = AudioPage(list_devices=_raising)
    qtbot.addWidget(page)
    page.initializePage()  # must NOT raise
    assert not page.isComplete()  # degrades to the no-device state
    assert page.no_device_alert.message() == "No speakers or headphones found."


def test_default_device_list_excludes_unflagged_device_named_loopback(monkeypatch):
    """Capture opens only devices PyAudioWPatch flags ``isLoopbackDevice``;
    a microphone or virtual input whose name merely contains "Loopback" must
    not be offered, or picking it fails at pipeline start. The default
    output's loopback carries ``"default": True``.

    Seeded defect: restore the name-substring match -> the mic is listed.
    """
    import sys
    import types

    from yamabiko.wizard_pages.audio import _default_list_devices

    infos = [
        {"name": "Loopback Mic (Virtual Audio)", "isLoopbackDevice": False},
        {"name": "Speakers (Realtek) [Loopback]", "isLoopbackDevice": True},
    ]

    class _FakePA:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_device_count(self):
            return len(infos)

        def get_device_info_by_index(self, i):
            return infos[i]

        def get_default_wasapi_loopback(self):
            return {"index": 1, **infos[1]}

    monkeypatch.setitem(
        sys.modules, "pyaudiowpatch", types.SimpleNamespace(PyAudio=_FakePA),
    )

    assert _default_list_devices() == [
        {"index": 1, "name": "Speakers (Realtek) [Loopback]", "default": True},
    ]


def test_default_device_lookup_failure_still_lists_devices(monkeypatch):
    """A default output without a loopback analogue makes PyAudioWPatch
    raise LookupError; the devices must still be listed, just unmarked.

    Seeded defect: call get_default_wasapi_loopback() outside its own
    try -> the whole enumeration falls into the outer except -> [] -> red.
    """
    import sys
    import types

    from yamabiko.wizard_pages.audio import _default_list_devices

    class _FakePA:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_device_count(self):
            return 1

        def get_device_info_by_index(self, i):
            return {"name": "Speakers (Realtek) [Loopback]", "isLoopbackDevice": True}

        def get_default_wasapi_loopback(self):
            raise LookupError("no loopback analogue for the default output")

    monkeypatch.setitem(
        sys.modules, "pyaudiowpatch", types.SimpleNamespace(PyAudio=_FakePA),
    )

    assert _default_list_devices() == [
        {"index": 0, "name": "Speakers (Realtek) [Loopback]"},
    ]


def test_page_complete_when_device_selected(qtbot):
    from yamabiko.wizard_pages.audio import AudioPage

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    assert page.isComplete()


def test_selected_device_name_exposed(qtbot):
    from yamabiko.wizard_pages.audio import AudioPage

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    page.device_list.segment("Headphones (USB) [Loopback]").click()
    assert page.selected_device_name() == "Headphones (USB) [Loopback]"


def test_interim_device_prefill_wins_over_merged_config_on_initialize(
    qtbot, monkeypatch,
):
    """A cancelled-session pick wins when the lazy device list is built.

    Seeded defect: omit the interim-device preference in initializePage ->
    the merged-config device at index 0 remains selected -> red.
    """
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"audio": {"device": "Speakers (Realtek) [Loopback]"}},
    )
    from yamabiko.wizard_pages.audio import AudioPage

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.set_interim_device_name("  Headphones (USB) [Loopback]  ")
    page.initializePage()

    assert page.selected_device_name() == "Headphones (USB) [Loopback]"


def test_interim_device_prefill_keeps_zero_device_relaunch_complete(qtbot):
    """A cancelled pick remains usable when no device enumerates on relaunch.

    Seeded defect: consult only merged config in initializePage's zero-device
    branch -> the restored pick disappears and the page hard-blocks -> red.
    """
    from yamabiko.wizard_pages.audio import AudioPage

    page = AudioPage(list_devices=lambda: [])
    qtbot.addWidget(page)
    page.set_interim_device_name("Headphones (USB) [Loopback]")
    page.initializePage()

    assert page.isComplete()
    assert page.selected_device_name() == "Headphones (USB) [Loopback]"
    assert page.no_device_alert.message() == "Can't see any speakers or headphones right now."
    assert page.no_device_alert.help_text() == "Yamabiko keeps using Headphones (USB)."


def test_revisit_keeps_current_choice_instead_of_stale_interim_prefill(qtbot):
    """Changing a restored pick survives the page's clear-and-repopulate.

    Seeded defect: prefer the prior-session interim seed over the list's
    current in-session choice on revisit -> the old headphones pick returns
    -> red.
    """
    from yamabiko.wizard_pages.audio import AudioPage

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.set_interim_device_name("Headphones (USB) [Loopback]")
    page.initializePage()
    assert page.selected_device_name() == "Headphones (USB) [Loopback]"

    page.device_list.segment("Speakers (Realtek) [Loopback]").click()
    page.initializePage()

    assert page.selected_device_name() == "Speakers (Realtek) [Loopback]"


def test_audio_test_opens_stream_at_device_rate_float32(qtbot, monkeypatch):
    """HDMI loopbacks reject 16 kHz / int16 outright. The callback-based
    audio test must open at the device's native rate + channel count +
    paFloat32, matching the runtime capture pipeline."""
    import sys
    import types
    from yamabiko.wizard_pages.audio import AudioPage

    open_calls: list[dict] = []

    class _FakeStream:
        def start_stream(self): pass
        def stop_stream(self): pass
        def close(self): pass

    class _FakePA:
        def get_device_count(self) -> int:
            return len(_fake_devices())

        def get_device_info_by_index(self, idx: int) -> dict:
            return {
                "name": _fake_devices()[idx]["name"],
                "defaultSampleRate": 48000.0,
                "maxInputChannels": 2,
            }

        def open(self, **kwargs):
            open_calls.append(kwargs)
            return _FakeStream()

        def terminate(self): pass

    _pa_sentinel = object()
    fake_pyaudio = types.SimpleNamespace(
        PyAudio=_FakePA,
        paFloat32=_pa_sentinel,
        paContinue=0,
    )
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", fake_pyaudio)

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    page._run_audio_test()

    # The stream open call must use device-native rate + channels + float32.
    assert len(open_calls) == 1
    rec = open_calls[0]
    assert rec["rate"] == 48000
    assert rec["channels"] == 2
    assert rec["format"] is _pa_sentinel
    assert rec.get("input") is True
    assert "stream_callback" in rec


def _install_fake_pyaudio(monkeypatch) -> None:
    import sys
    import types

    class _FakeStream:
        def start_stream(self): pass
        def stop_stream(self): pass
        def close(self): pass

    class _FakePA:
        def get_device_count(self) -> int:
            return len(_fake_devices())

        def get_device_info_by_index(self, idx: int) -> dict:
            return {
                "name": _fake_devices()[idx]["name"],
                "defaultSampleRate": 48000.0,
                "maxInputChannels": 2,
            }

        def open(self, **kwargs):
            return _FakeStream()

        def terminate(self): pass

    monkeypatch.setitem(
        sys.modules, "pyaudiowpatch",
        types.SimpleNamespace(PyAudio=_FakePA, paFloat32=object(), paContinue=0),
    )


def test_audio_test_finish_timer_dies_with_destroyed_page(
    qapp, monkeypatch, capfd,
):
    """The 5 s finish timer must be a CHILD of the page: if the user starts
    the audio test and closes the wizard inside the window, the timer must die
    with the page instead of firing `_finish_test` on the deleted C++ object
    (libshiboken RuntimeError through the event loop).

    Explicit excepthook recorder + stderr scan: relying only on pytest-qt's
    implicit teardown capture would silently turn this into a Liar if
    qt_no_exception_capture were ever set. Same recorder pattern as
    test_dwm_suppression.py::test_show_then_quick_close_kills_pending_chrome_retry_timer.

    Seeded defect: revert the parented timer to a free-standing
    QTimer.singleShot -> the timer outlives the page, fires on the deleted
    object, and the hook/stderr asserts below go red.
    """
    import sys as _sys

    from PySide6.QtCore import QEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from yamabiko.wizard_pages.audio import AudioPage

    _install_fake_pyaudio(monkeypatch)
    monkeypatch.setattr(AudioPage, "_TEST_DURATION_MS", 50)

    page = AudioPage(list_devices=_fake_devices)
    page.initializePage()
    page._run_audio_test()

    hook_errors: list[tuple] = []
    old_hook = _sys.excepthook
    _sys.excepthook = lambda *exc: hook_errors.append(exc)
    try:
        # Destroy the page BEFORE the finish deadline (flush the
        # deferred delete so the C++ object is truly gone), then pump
        # well past the deadline so a surviving timer fires now,
        # inside the capture window.
        page.deleteLater()
        del page
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QTest.qWait(200)
    finally:
        _sys.excepthook = old_hook

    err = capfd.readouterr().err
    assert not hook_errors, (
        f"audio-test finish timer fired on the destroyed page: {hook_errors}"
    )
    assert "RuntimeError" not in err and "already deleted" not in err, (
        f"PySide6 reported a deleted-object access on stderr:\n{err}"
    )


def _install_recording_pyaudio(monkeypatch, *, device_info_exc=None):
    """Fake pyaudiowpatch whose PA/stream instances record lifecycle
    calls and expose the captured stream callback, so tests can assert
    resource release and drive audio frames through the RMS path."""
    import sys
    import types

    rec = types.SimpleNamespace(
        open_calls=0,
        terminate_calls=0,
        stop_stream_calls=0,
        close_calls=0,
        callback=None,
    )

    class _RecStream:
        def start_stream(self):
            pass

        def stop_stream(self):
            rec.stop_stream_calls += 1

        def close(self):
            rec.close_calls += 1

    class _RecPA:
        def get_device_count(self) -> int:
            return len(_fake_devices())

        def get_device_info_by_index(self, idx: int) -> dict:
            if device_info_exc is not None:
                raise device_info_exc
            return {
                "name": _fake_devices()[idx]["name"],
                "defaultSampleRate": 48000.0,
                "maxInputChannels": 2,
            }

        def open(self, **kwargs):
            rec.open_calls += 1
            rec.callback = kwargs.get("stream_callback")
            return _RecStream()

        def terminate(self):
            rec.terminate_calls += 1

    monkeypatch.setitem(
        sys.modules, "pyaudiowpatch",
        types.SimpleNamespace(PyAudio=_RecPA, paFloat32=object(), paContinue=0),
    )
    return rec


def test_audio_test_device_info_error_terminates_pa(qtbot, monkeypatch):
    """The get_device_info_by_index failure path must terminate() the
    just-created PyAudio instance -- previously it returned without doing so,
    leaking the PortAudio COM instance on every failed Test click (asymmetric
    with the pa.open() failure path, which terminates).

    Seeded defect: remove pa.terminate() from the device-info except block in
    _run_audio_test -> terminate_calls stays 0 -> red.
    """
    from yamabiko.wizard_pages.audio import AudioPage

    rec = _install_recording_pyaudio(
        monkeypatch, device_info_exc=OSError("device vanished"),
    )
    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    page._run_audio_test()

    assert rec.terminate_calls == 1, (
        "PyAudio instance leaked on the device-info failure path"
    )
    assert page.test_button.isEnabled()
    assert page.result_label.text() == "Couldn't listen to this device. Pick another one."
    assert page.result_mark.state() == "paused"


def test_audio_test_survives_pyaudio_constructor_failure(qtbot, monkeypatch):
    """When PyAudio() itself raises (Windows Audio service stopped), there is
    no instance to terminate -- the failure path must not NameError on the
    unbound ``pa`` and must degrade to the failure status with the button
    re-enabled.

    Seeded defect: reference ``pa`` unconditionally in the device-info except
    block (terminate without the None guard) -> NameError propagates out of
    the clicked slot -> red.
    """
    import sys
    import types

    from yamabiko.wizard_pages.audio import AudioPage

    class _ExplodingPA:
        def __init__(self):
            raise OSError("Windows Audio service is not running")

    monkeypatch.setitem(
        sys.modules, "pyaudiowpatch",
        types.SimpleNamespace(
            PyAudio=_ExplodingPA, paFloat32=object(), paContinue=0,
        ),
    )

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    page._run_audio_test()  # must NOT raise

    assert page.test_button.isEnabled()
    assert page.result_label.text() == "Couldn't listen to this device. Pick another one."


def test_audio_test_retry_resolves_selected_name_in_fresh_pyaudio_list(
    qtbot, monkeypatch,
):
    """A retry after one endpoint fails must not reuse cached PA indexes.

    PortAudio device indexes belong to the device list created by that
    ``PyAudio`` instance.  The wizard's device list can outlive that instance,
    so its cached index may be stale by the next Test click.  This models the live
    release failure: the first endpoint rejects open, the user selects a
    working endpoint, and the new PyAudio list has different numeric indexes.

    Seeded defect: open ``input_device_index`` from the enumeration's cached
    index (40/41) instead of resolving the selected name in the new
    PyAudio instance -> both clicks fail on the stale indexes and the second
    stream never starts.
    """
    import sys
    import types

    from yamabiko.wizard_pages.audio import AudioPage

    current_devices = [
        {
            "index": 0,
            "name": "Unavailable HDMI [Loopback]",
            "defaultSampleRate": 48000.0,
            "maxInputChannels": 2,
        },
        {
            "index": 1,
            "name": "Speakers (Working) [Loopback]",
            "defaultSampleRate": 48000.0,
            "maxInputChannels": 2,
        },
    ]
    opened_indexes: list[int] = []
    streams: list[object] = []

    class _Stream:
        def __init__(self):
            self.started = False

        def start_stream(self):
            self.started = True

        def stop_stream(self):
            pass

        def close(self):
            pass

    class _FreshPA:
        def get_device_count(self):
            return len(current_devices)

        def get_device_info_by_index(self, idx: int):
            if not 0 <= idx < len(current_devices):
                raise OSError(f"invalid device index {idx}")
            return current_devices[idx]

        def open(self, **kwargs):
            idx = kwargs["input_device_index"]
            opened_indexes.append(idx)
            if idx == 0:
                raise OSError("device unavailable")
            stream = _Stream()
            streams.append(stream)
            return stream

        def terminate(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "pyaudiowpatch",
        types.SimpleNamespace(
            PyAudio=_FreshPA,
            paFloat32=object(),
            paContinue=0,
        ),
    )

    # These are the indexes cached by the earlier enumeration.  Neither is
    # valid in the fresh PyAudio list used by the Test button.
    page = AudioPage(list_devices=lambda: [
        {"index": 40, "name": current_devices[0]["name"]},
        {"index": 41, "name": current_devices[1]["name"]},
    ])
    qtbot.addWidget(page)
    page.initializePage()

    page._run_audio_test()
    assert page.test_button.isEnabled()
    assert page.result_label.text() == "Couldn't listen to this device. Pick another one."

    page.device_list.segment("Speakers (Working) [Loopback]").click()
    page._run_audio_test()

    assert page.result_label.isHidden()  # running: no result line yet
    assert not page.test_button.isEnabled()
    assert opened_indexes == [0, 1]
    assert len(streams) == 1 and streams[0].started is True
    page.cleanupPage()


def test_audio_test_cleanup_page_releases_stream_and_pa(qtbot, monkeypatch):
    """Back navigation mid-test (Qt calls cleanupPage) must stop + close the
    test stream and terminate the PA instance -- the page outlives Back, so
    destruction-time release never fires on that path.

    Seeded defect: delete AudioPage.cleanupPage -> the QWizardPage base no-op
    runs instead -> all three counters stay 0 -> red.
    """
    from yamabiko.wizard_pages.audio import AudioPage

    rec = _install_recording_pyaudio(monkeypatch)
    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    page._run_audio_test()
    assert rec.open_calls == 1

    page.cleanupPage()

    assert rec.stop_stream_calls == 1
    assert rec.close_calls == 1
    assert rec.terminate_calls == 1
    # Idempotence: a second release path (late finish timer, repeated
    # Back) must not double-stop/double-terminate.
    page.cleanupPage()
    assert rec.terminate_calls == 1
    assert rec.close_calls == 1


def test_audio_test_page_destruction_mid_test_releases_stream_and_pa(
    qapp, monkeypatch,
):
    """Closing the wizard inside the 5 s test window destroys the page -- the
    page-parented finish timer dies with it, so _finish_test never runs. The
    destroyed-signal hook must release the PyAudio stream + instance instead;
    without it the WASAPI callback thread and PortAudio COM instance survive
    for the rest of the process.

    Seeded defect: drop the destroyed-signal release wiring from
    AudioPage.__init__ -> counters stay 0 after destruction -> red.
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from yamabiko.wizard_pages.audio import AudioPage

    rec = _install_recording_pyaudio(monkeypatch)
    # Keep the production-length window: destruction must beat the timer.
    page = AudioPage(list_devices=_fake_devices)
    page.initializePage()
    page._run_audio_test()
    assert rec.open_calls == 1

    page.deleteLater()
    del page
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QTest.qWait(50)

    assert rec.terminate_calls == 1, (
        "PyAudio instance outlived the destroyed page"
    )
    assert rec.stop_stream_calls == 1
    assert rec.close_calls == 1


def test_audio_test_reports_no_audio_when_stream_is_silent(qtbot, monkeypatch):
    """A run where no frame ever carries signal (wrong device picked, nothing
    playing) must say so -- the completion message must not claim "audio is
    working" unconditionally, which would defeat the test's purpose.

    Seeded defect: make _finish_test set the success string without the
    session-max-RMS branch -> red.
    """
    from yamabiko.wizard_pages.audio import AudioPage

    _install_recording_pyaudio(monkeypatch)
    monkeypatch.setattr(AudioPage, "_TEST_DURATION_MS", 50)

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    page._run_audio_test()  # the fake stream never invokes the callback

    qtbot.waitUntil(lambda: page.test_button.isEnabled(), timeout=2000)
    assert page.result_label.text() == (
        "No sound picked up. Check that something is playing on this device."
    ), f"silent test run claimed success: {page.result_label.text()!r}"
    assert page.result_mark.state() == "paused"


# ---------------------------------------------------------------------------
# Evidence-erasing swallows in the release closure (2026-07-01 logging
# audit, Task 9): _make_test_audio_release's two broad excepts must KEEP
# swallowing (release runs at page destruction and must never raise into
# Qt teardown) but now leave a WARNING with the traceback in the log.
# Direct closure invocation per the charter's 3.9 rule.
# ---------------------------------------------------------------------------


def test_test_audio_release_stream_close_failure_logs_warning_and_still_terminates_pa(
    caplog,
):
    """A stream stop/close failure inside the release closure must log a
    WARNING (with traceback) and STILL fall through to pa.terminate() --
    the bare ``except Exception: pass`` previously erased the only
    evidence of a PortAudio stream dying under release.

    Seeded defect: re-silence the stream except in
    _make_test_audio_release (revert to ``except Exception: pass``) ->
    no WARNING record -> red.
    """
    import logging

    from yamabiko.wizard_pages.audio import _make_test_audio_release

    terminated: list[bool] = []

    class _BoomStream:
        def stop_stream(self):
            raise OSError("PortAudio stream already invalid")

        def close(self):
            pass

    class _PA:
        def terminate(self):
            terminated.append(True)

    release = _make_test_audio_release(_BoomStream(), _PA())
    with caplog.at_level(
        logging.WARNING, logger="yamabiko.wizard_pages.audio",
    ):
        release()  # must NOT raise

    # Swallow semantics KEPT: the PA instance is still terminated.
    assert terminated == [True], (
        "a stream stop/close failure must not skip pa.terminate()"
    )
    recs = [
        r for r in caplog.records
        if r.name == "yamabiko.wizard_pages.audio"
        and r.levelno == logging.WARNING
    ]
    assert recs, "swallowed stream stop/close failure left no WARNING evidence"
    assert recs[0].exc_info and isinstance(recs[0].exc_info[1], OSError), (
        "the WARNING must carry the swallowed traceback (exc_info=True)"
    )


def test_test_audio_release_pa_terminate_failure_logs_warning_without_raising(
    caplog,
):
    """A pa.terminate() failure inside the release closure must log a
    WARNING (with traceback) and not raise -- release is wired to the
    page's ``destroyed`` signal, where an escaping exception lands in Qt
    teardown; the previous bare swallow kept that safety but erased the
    evidence of a leaked PortAudio COM instance.

    Seeded defect: re-silence the terminate except in
    _make_test_audio_release (revert to ``except Exception: pass``) ->
    no WARNING record -> red.
    """
    import logging

    from yamabiko.wizard_pages.audio import _make_test_audio_release

    closed: list[bool] = []

    class _Stream:
        def stop_stream(self):
            pass

        def close(self):
            closed.append(True)

    class _BoomPA:
        def terminate(self):
            raise OSError("PortAudio instance already torn down")

    release = _make_test_audio_release(_Stream(), _BoomPA())
    with caplog.at_level(
        logging.WARNING, logger="yamabiko.wizard_pages.audio",
    ):
        release()  # must NOT raise

    # Swallow semantics KEPT: the stream was released before the failure.
    assert closed == [True]
    recs = [
        r for r in caplog.records
        if r.name == "yamabiko.wizard_pages.audio"
        and r.levelno == logging.WARNING
    ]
    assert recs, "swallowed pa.terminate() failure left no WARNING evidence"
    assert recs[0].exc_info and isinstance(recs[0].exc_info[1], OSError), (
        "the WARNING must carry the swallowed traceback (exc_info=True)"
    )


def test_audio_test_lights_the_meter_then_reports_picking_up_sound(qtbot, monkeypatch):
    """Drives a real float32 block through the captured stream callback:
    the meter lights while the test runs; when it ends, the result line
    shows the check with "Picking up sound." beside the meter holding the
    test's loudest level (Decision 140, screenshots 03/04).

    Seeded defects: delete the finish-timer start -> waitUntil times out;
    drop the session-max tracking from the audio callback -> "No sound
    picked up." -> red; drop the hold in _finish_test (rest the meter at 0,
    or leave it where the smoothing decayed to) -> 0 cells -> red.
    """
    import numpy as np

    from yamabiko.wizard_pages.audio import AudioPage

    rec = _install_recording_pyaudio(monkeypatch)
    monkeypatch.setattr(AudioPage, "_TEST_DURATION_MS", 1000)

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    page._run_audio_test()

    # One 480-frame stereo block at half amplitude (RMS 0.5 >> floor).
    assert rec.callback is not None, "stream callback was not captured"
    block = np.full(960, 0.5, dtype=np.float32)
    rec.callback(block.tobytes(), 480, None, 0)

    qtbot.waitUntil(lambda: page.meter.lit_cells() > 0, timeout=2000)
    qtbot.waitUntil(lambda: page.test_button.isEnabled(), timeout=3000)
    assert page.result_label.text() == "Picking up sound."
    assert page.result_mark.state() == "done"
    # The first 100 ms tick showed the loudest level, 0.35 x 0.5 RMS x 4 =
    # 0.7 of the meter (22 of 32 cells); later ticks decay toward 0.
    assert page.meter.lit_cells() == 22
    assert not page._meter_timer.isActive()


# ---------------------------------------------------------------------------
# P2: zero-loopback machines vs the FORCED upgrade wizard. An RDP session
# enumerates no WASAPI loopback endpoints, and the upgrade path forces every
# v0.1.0 user through the wizard (model pins changed) -- a hard block on
# device count bricked a previously working install in the cancel/exit-42
# loop even though user_overrides.yaml already held a working device name.
# ---------------------------------------------------------------------------


def test_zero_devices_with_saved_config_device_completes_with_keeping_warning(
    qtbot, monkeypatch,
):
    """Zero enumerated devices + a saved non-default audio.device in the
    merged config -> the page completes, and the page says which saved
    device it keeps using.

    Seeded defect: revert isComplete to the bare device-count check -> the
    page stays blocked -> red.
    """
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"audio": {"device": "Speakers (Realtek) [Loopback]"}},
    )
    from yamabiko.wizard_pages.audio import AudioPage

    page = AudioPage(list_devices=lambda: [])
    qtbot.addWidget(page)
    page.initializePage()

    assert page.isComplete(), (
        "a saved-device install must be able to finish the forced wizard "
        "even when no loopback devices enumerate right now"
    )
    assert page.no_device_alert.message() == "Can't see any speakers or headphones right now."
    assert page.no_device_alert.help_text() == "Yamabiko keeps using Speakers (Realtek)."
    # The list stays empty: accept() writes nothing for an empty pick, so
    # the saved override survives untouched.
    assert page.selected_device_name() == ""


def test_zero_devices_with_shipped_default_device_stays_blocked(
    qtbot, monkeypatch,
):
    """A true first run (audio.device still the shipped 'default') keeps the
    hard block -- there is no saved pick to fall back to.

    Seeded defect: treat 'default' as a saved device in _saved_audio_device
    -> the page completes -> red.
    """
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"audio": {"device": "default"}},
    )
    from yamabiko.wizard_pages.audio import AudioPage

    page = AudioPage(list_devices=lambda: [])
    qtbot.addWidget(page)
    page.initializePage()

    assert not page.isComplete()
    assert page.no_device_alert.message() == "No speakers or headphones found."


def test_zero_devices_with_config_read_failure_stays_blocked(qtbot, monkeypatch):
    """A config-read failure fails CLOSED: no claimed saved device, page
    blocked (today's behavior), and initializePage must not raise.

    Seeded defect: let _saved_audio_device propagate the exception -> this
    test errors -> red.
    """
    def _boom():
        raise OSError("user_overrides.yaml unreadable")

    monkeypatch.setattr("yamabiko.config.load_full_config", _boom)
    from yamabiko.wizard_pages.audio import AudioPage

    page = AudioPage(list_devices=lambda: [])
    qtbot.addWidget(page)
    page.initializePage()  # must NOT raise

    assert not page.isComplete()
    assert page.no_device_alert.message() == "No speakers or headphones found."


def test_devices_reappearing_on_revisit_drop_the_saved_device_leg(
    qtbot, monkeypatch,
):
    """Back/Next after plugging in audio: the enumerated-device path takes
    over -- the zero-device warning is cleared and completion flows from the
    list selection, not the saved-device fallback.

    Seeded defect: keep the stale _saved_device / warning text across the
    re-entry -> the zero-device alert stays -> red.
    """
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"audio": {"device": "Speakers (Realtek) [Loopback]"}},
    )
    from yamabiko.wizard_pages.audio import AudioPage

    devices: list = []
    page = AudioPage(list_devices=lambda: list(devices))
    qtbot.addWidget(page)
    page.initializePage()
    assert page.isComplete()  # saved-device leg while nothing enumerates

    devices.extend(_fake_devices())
    page.initializePage()  # revisit with devices present

    assert page._saved_device is None
    assert page.no_device_alert.isHidden()
    assert _row_names(page) == ["Speakers (Realtek)", "Headphones (USB)"]
    assert page.isComplete()
    assert page.selected_device_name() == "Speakers (Realtek) [Loopback]"


# ---------------------------------------------------------------------------
# Devices DO enumerate + a saved config device: the list must preselect the
# saved pick. The upgrade path forces every completed install back through
# the wizard (model pins changed), and accept() writes the CURRENT list
# choice -- an unseeded list on its first row silently overwrites a working
# device (e.g. VB-Cable) with whatever enumerates first.
# ---------------------------------------------------------------------------


def test_enumerated_devices_preselect_the_saved_config_device(
    qtbot, monkeypatch,
):
    """A saved non-default audio.device that is among the enumerated names
    must be preselected (here at a NON-zero index), so an untouched
    click-through round-trips the saved pick.

    Seeded defect: drop the saved-device seeding from initializePage's
    devices-present branch -> the first row stays selected -> red.
    """
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"audio": {"device": "Headphones (USB) [Loopback]"}},
    )
    from yamabiko.wizard_pages.audio import AudioPage

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()

    assert page.device_list.value() == "Headphones (USB) [Loopback]", (
        "saved device enumerates second but was not preselected"
    )
    assert page.selected_device_name() == "Headphones (USB) [Loopback]"
    assert page.isComplete()


def test_enumerated_devices_keep_first_selection_when_saved_device_absent(
    qtbot, monkeypatch,
):
    """A saved device that no longer enumerates (unplugged VB-Cable) keeps
    today's behavior: first device selected, page complete, no zero-device
    guidance.

    Seeded defect: match the saved device loosely (e.g. substring) or
    force-select a nonexistent entry -> index/name drift -> red.
    """
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"audio": {"device": "CABLE Output (VB-Audio) [Loopback]"}},
    )
    from yamabiko.wizard_pages.audio import AudioPage

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()

    assert page.device_list.value() == "Speakers (Realtek) [Loopback]"
    assert page.selected_device_name() == "Speakers (Realtek) [Loopback]"
    assert page.isComplete()
    assert page.no_device_alert.isHidden()


def test_more_than_five_devices_scroll_inside_a_five_row_list(qtbot):
    """L8: the list shows at most five 36 px rows; more devices scroll
    inside the list, so the step never grows past the 600 px window.

    Seeded defect: size the list area to all its rows -> 7 rows tall -> red.
    """
    from yamabiko.wizard_pages.audio import AudioPage

    devices = [{"index": i, "name": f"Output {i} [Loopback]"} for i in range(7)]
    page = AudioPage(list_devices=lambda: devices)
    qtbot.addWidget(page)
    page.initializePage()

    assert page.device_area.maximumHeight() == 5 * 36 + 4 * 2 + 6  # rows, gaps, 3 px rims
    assert page.device_list.minimumHeight() == 7 * 36 + 6 * 2 + 4  # the edge is the area's


def test_look_again_lists_devices_that_appeared(qtbot):
    """Flow Q10f: with no device, "Look again" enumerates again; a device
    that appeared is listed and the step completes.

    Seeded defect: leave the button unconnected -> the step stays empty.
    """
    from yamabiko.wizard_pages.audio import AudioPage

    devices: list = []
    page = AudioPage(list_devices=lambda: list(devices))
    qtbot.addWidget(page)
    page.initializePage()
    assert not page.isComplete()

    devices.extend(_fake_devices())
    page.look_again_button.click()

    assert page.isComplete()
    assert page.no_device_alert.isHidden()
    assert page.selected_device_name() == "Speakers (Realtek) [Loopback]"


def test_picking_another_device_clears_the_result_and_the_held_meter(qtbot, monkeypatch):
    """Decision 66: a result belongs to the device it was measured on, so
    picking another row hides "Picking up sound." and turns off the level
    the meter held with it (Decision 140).

    Seeded defect: drop the _show_result(None) call from _on_device_changed
    -> the old result stays -> red.
    """
    import numpy as np

    from yamabiko.wizard_pages.audio import AudioPage

    rec = _install_recording_pyaudio(monkeypatch)
    monkeypatch.setattr(AudioPage, "_TEST_DURATION_MS", 300)

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    page._run_audio_test()
    rec.callback(np.full(960, 0.5, dtype=np.float32).tobytes(), 480, None, 0)
    qtbot.waitUntil(lambda: page.result_label.text() == "Picking up sound.", timeout=3000)
    assert page.meter.lit_cells() > 0

    page.device_list.segment("Headphones (USB) [Loopback]").click()

    assert page.result_label.isHidden()
    assert page.meter.lit_cells() == 0


def test_picking_another_device_mid_test_stops_the_test_without_a_verdict(qtbot, monkeypatch):
    """Decision 66: a pick during the 5 s test stops that test (both
    timers, the open stream), so the first device's verdict never shows
    beside the new pick; the meter goes dark and Test is clickable again.

    Seeded defect: drop the _stop_test() call from _on_device_changed ->
    the finish timer still fires and shows "Picking up sound." -> red.
    """
    import numpy as np

    from yamabiko.wizard_pages.audio import AudioPage

    rec = _install_recording_pyaudio(monkeypatch)
    monkeypatch.setattr(AudioPage, "_TEST_DURATION_MS", 300)

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    page._run_audio_test()
    rec.callback(np.full(960, 0.5, dtype=np.float32).tobytes(), 480, None, 0)

    page.device_list.segment("Headphones (USB) [Loopback]").click()
    qtbot.wait(2 * AudioPage._TEST_DURATION_MS)

    assert page.result_label.isHidden()
    assert page.meter.lit_cells() == 0
    assert page.test_button.isEnabled()
    assert (rec.close_calls, rec.terminate_calls) == (1, 1)


def test_leaving_the_step_mid_test_turns_the_live_meter_dark(qtbot, monkeypatch):
    """Decision 140: Back or a rail jump during the test cuts it short; a
    test cut short has no result, so the meter it was driving goes dark and
    no verdict shows later.

    Seeded defect: cleanupPage stops the test but keeps the meter as it was
    -> lit cells stay -> red.
    """
    import numpy as np

    from yamabiko.wizard_pages.audio import AudioPage

    rec = _install_recording_pyaudio(monkeypatch)
    monkeypatch.setattr(AudioPage, "_TEST_DURATION_MS", 300)

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    page._run_audio_test()
    rec.callback(np.full(960, 0.5, dtype=np.float32).tobytes(), 480, None, 0)
    qtbot.waitUntil(lambda: page.meter.lit_cells() > 0, timeout=2000)

    page.cleanupPage()
    qtbot.wait(2 * AudioPage._TEST_DURATION_MS)

    assert page.meter.lit_cells() == 0
    assert page.result_label.isHidden()


def test_entering_the_step_again_mid_test_stops_that_test(qtbot, monkeypatch):
    """A rail jump back past Sound cleans up only the page left (L2), so a
    test started on Sound still runs when Next enters the step again, and
    initializePage enables Test: a second run would take over the first
    run's timers and release, leaving its stream open. Entering the step
    stops a running test first; a test cut short has no verdict.

    Seeded defect: drop the stop_test() call from initializePage -> the
    first stream stays open and its verdict shows -> red.
    """
    import numpy as np

    from yamabiko.wizard_pages.audio import AudioPage

    rec = _install_recording_pyaudio(monkeypatch)
    monkeypatch.setattr(AudioPage, "_TEST_DURATION_MS", 300)

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    page._run_audio_test()
    rec.callback(np.full(960, 0.5, dtype=np.float32).tobytes(), 480, None, 0)

    page.initializePage()

    assert (rec.stop_stream_calls, rec.close_calls, rec.terminate_calls) == (1, 1, 1)
    assert page.test_button.isEnabled()
    qtbot.wait(2 * AudioPage._TEST_DURATION_MS)
    assert page.result_label.isHidden()
    assert page.meter.lit_cells() == 0


def test_a_finished_test_keeps_its_result_and_held_meter_across_back_and_return(
    qtbot, monkeypatch,
):
    """Decision 140: a finished test's result line and the level the meter
    holds with it stay when the user goes Back and returns to the step
    (cleanupPage, then initializePage), until another device is picked.

    Seeded defect: cleanupPage clears the result whether or not a test was
    running -> red.
    """
    import numpy as np

    from yamabiko.wizard_pages.audio import AudioPage

    rec = _install_recording_pyaudio(monkeypatch)
    monkeypatch.setattr(AudioPage, "_TEST_DURATION_MS", 300)

    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    page._run_audio_test()
    rec.callback(np.full(960, 0.5, dtype=np.float32).tobytes(), 480, None, 0)
    qtbot.waitUntil(lambda: page.result_label.text() == "Picking up sound.", timeout=3000)
    held = page.meter.lit_cells()
    assert held > 0

    page.cleanupPage()
    page.initializePage()

    assert not page.result_label.isHidden()
    assert page.result_label.text() == "Picking up sound."
    assert page.meter.lit_cells() == held


def test_picking_a_row_below_the_fifth_scrolls_it_into_view(qtbot):
    """Decision 145: arrow keys move the choice (valueChanged) without
    scrolling the list, and QScrollArea follows focus only on Tab; a pick
    below the fifth row must bring that row into view.

    Seeded defect: drop the _scroll_to_selection call from
    _on_device_changed -> the list stays at the top -> red.
    """
    from yamabiko.wizard_pages.audio import AudioPage

    devices = [{"index": i, "name": f"Output {i} [Loopback]"} for i in range(7)]
    page = AudioPage(list_devices=lambda: devices)
    qtbot.addWidget(page)
    page.initializePage()
    page.show()
    qtbot.waitExposed(page)
    bar = page.device_area.verticalScrollBar()
    qtbot.waitUntil(lambda: bar.maximum() > 0, timeout=2000)
    assert bar.value() == 0  # the first row is the choice

    page.device_list.segment("Output 6 [Loopback]").click()

    assert bar.value() == bar.maximum()


def _render(widget):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QImage, QRegion
    from PySide6.QtWidgets import QWidget

    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    widget.render(image, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return image


def test_a_scrolling_device_list_keeps_its_frame_closed_with_the_scroll_bar_inside(qtbot):
    """Visual check M2: with more than five devices the rows scroll inside
    a closed frame, as the mockup's .devs draws it: the 1 px edge, bottom
    included, stays put at either end of the scroll, the scroll bar sits
    within the edge, and the rows stay on the field tone.

    Seeded defects: let the scrolled list paint the frame inside a frameless
    scroll area (the visual check's build) -> the frame scrolls with the
    rows, its bottom edge out of view, and the bar sits outside it -> red;
    keep the background fill QScrollArea.setWidget turns on for the list ->
    the rows sit on the palette's gray -> red.
    """
    from PySide6.QtCore import QPoint, QRect

    from yamabiko.overlay_app.theme import MIST
    from yamabiko.overlay_app.window_chrome import apply_window_qss
    from yamabiko.wizard_pages.audio import AudioPage

    devices = [{"index": i, "name": f"Output {i} [Loopback]"} for i in range(7)]
    page = AudioPage(list_devices=lambda: devices)
    qtbot.addWidget(page)
    apply_window_qss(page, MIST)  # the setup window's QSS
    page.initializePage()
    page.resize(620, 480)
    page.show()
    qtbot.waitExposed(page)
    bar = page.device_area.verticalScrollBar()
    qtbot.waitUntil(lambda: bar.isVisible() and bar.maximum() > 0, timeout=2000)
    box = page.device_area.geometry()

    for value in (bar.minimum(), bar.maximum()):
        bar.setValue(value)
        image = _render(page)
        edge = image.pixelColor(box.center().x(), box.bottom()).name()
        assert edge == MIST["edge"], f"no bottom edge at scroll {value}: {edge}"
    # Scrolled to the end every visible row is unlit: the middle is the field.
    assert image.pixelColor(box.center()).name() == MIST["field"]
    bar_box = QRect(bar.mapTo(page, QPoint(0, 0)), bar.size())
    assert box.adjusted(1, 1, -1, -1).contains(bar_box), (bar_box, box)


def _keyboard_ring_on(row) -> bool:
    """Whether the lit row paints its keyboard ring: the on-accent inset
    line inside the accent fill, left of the label."""
    from yamabiko.overlay_app.theme import MIST

    image = _render(row)
    return MIST["on_accent"] in {image.pixelColor(2, y).name() for y in range(6, row.height() - 6)}


def test_the_row_the_test_button_hands_its_focus_to_shows_no_ring_until_tab(
    qtbot, monkeypatch,
):
    """Visual check M1a: clicking "Test for 5 seconds" focuses the button and
    the test disables it, so Qt hands the focus on to the selected row, with
    a Tab reason. No key moved it: no keyboard ring. Shift+Tab and Tab back
    onto the row are keyboard use: the on-accent ring shows.

    Seeded defect: KeyboardRingMixin takes every Tab-reason focus for
    keyboard use (its old rule) -> the row rings after the click -> red.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from yamabiko.wizard_pages.audio import AudioPage

    _install_recording_pyaudio(monkeypatch)
    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    page.show()
    page.activateWindow()
    assert QTest.qWaitForWindowActive(page)
    row = page.device_list.segment(page.selected_device_name())

    QTest.mouseClick(page.test_button, Qt.LeftButton)

    assert row.hasFocus(), QApplication.focusWidget()
    assert not _keyboard_ring_on(row)
    QTest.keyClick(row, Qt.Key_Backtab)
    QTest.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
    assert row.hasFocus(), QApplication.focusWidget()
    assert _keyboard_ring_on(row)


def test_the_row_a_space_on_the_test_button_hands_its_focus_to_shows_the_ring(
    qtbot, monkeypatch,
):
    """Review of fix pass 3, Important 1: Tab to "Test for 5 seconds" and
    Space start the test, which disables the button, so Qt hands the focus on
    to the selected row. A key started it: the row shows the on-accent ring,
    so the keyboard user still sees where the focus is.

    Seeded defect: ring only while Qt's WA_KeyboardFocusChange is set (fix
    pass 3's rule; Qt 6.10.1+ clears it before this hand-over) -> no ring ->
    red."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from yamabiko.wizard_pages.audio import AudioPage

    _install_recording_pyaudio(monkeypatch)
    page = AudioPage(list_devices=_fake_devices)
    qtbot.addWidget(page)
    page.initializePage()
    page.show()
    page.activateWindow()
    assert QTest.qWaitForWindowActive(page)
    row = page.device_list.segment(page.selected_device_name())
    for _press in range(6):
        if page.test_button.hasFocus():
            break
        QTest.keyClick(QApplication.focusWidget() or page, Qt.Key_Tab)
    assert page.test_button.hasFocus(), QApplication.focusWidget()

    QTest.keyClick(page.test_button, Qt.Key_Space)

    assert row.hasFocus(), QApplication.focusWidget()
    assert _keyboard_ring_on(row)
