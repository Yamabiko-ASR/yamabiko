"""Wizard step "Sound": the output Yamabiko listens to, and a 5-second test.

Device enumeration is injected via ``list_devices`` so tests don't need
real audio hardware; a live WASAPI query is the production default (Settings
reuses it). The list names each loopback endpoint by its output (the
PyAudio name without " [Loopback]") and marks the Windows default output;
the raw PyAudio name is what the wizard persists (``audio.device``).

The 5-second test is a loopback CAPTURE check, not a playback: the user
already hears the audio through their own speakers, so the 32-cell meter
only confirms frames are flowing, and the result line says whether any
sound arrived above the runtime's silence floor.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWizardPage,
)

# Single source of truth for the silence floor: the wizard's 5 s test meter and
# the runtime's idle re-open (capture/audio.py, issue #54) must use the SAME
# threshold, so the wizard can't tell a user "audio is working" at a level the
# runtime would later treat as silence. Imported from the dependency-free
# constants module (NOT capture.audio) so the wizard's import graph stays free
# of pyaudiowpatch -- the wizard guards PortAudio at call time and must not
# crash at import on a partial/broken bundle.
from yamabiko.capture._audio_constants import SILENCE_RMS_FLOOR as _SILENCE_RMS_FLOOR
from yamabiko.overlay_app.controls import mincho_font, ui_font
from yamabiko.overlay_app.segmented import SegmentOption, SegmentedControl
from yamabiko.overlay_app.theme import MIST
from yamabiko.wizard_pages.parts import Alert, LevelMeter, StateMark

_log = logging.getLogger(__name__)

PICKING_UP = "Picking up sound."
NO_SOUND = "No sound picked up. Check that something is playing on this device."
OPEN_FAILED = "Couldn't listen to this device. Pick another one."
NO_DEVICES = "No speakers or headphones found."
NO_DEVICES_HELP = "Connect one, then click Look again."
DEVICES_GONE = "Can't see any speakers or headphones right now."
SYSTEM_DEFAULT = "System default"

_LOOPBACK_SUFFIX = " [Loopback]"
# The segmented "list" variant: 36 px rows 2 px apart inside a 3 px rim
# (the scroll area's 1 px edge + the list's 2 px padding). Five rows show;
# more scroll (L8).
_ROW_H = 36
_ROW_GAP = 2
_LIST_RIM = 3
_MAX_ROWS = 5


def _display_name(name: str) -> str:
    """The output's name as Windows shows it: the PyAudio loopback name
    without its " [Loopback]" suffix (the error dialog's rule)."""
    return name.removesuffix(_LOOPBACK_SUFFIX)


def _list_height(rows: int) -> int:
    return rows * _ROW_H + (rows - 1) * _ROW_GAP + 2 * _LIST_RIM


def _make_test_audio_release(stream, pa) -> Callable[[], None]:
    """Build an idempotent closure that stops/closes a wizard test
    stream and terminates its PyAudio instance.

    The closure deliberately captures the PortAudio handles DIRECTLY
    and never references the page: it is connected to the page's
    ``destroyed`` signal, where (a) a bound-method slot whose receiver
    is the dying page is dropped by Qt before the signal fires
    (observed live on PySide6 6.11), and (b) touching any Qt member of
    the page would be a use-after-free. PortAudio handles are plain
    Python/C objects and stay valid.
    """
    released = [False]

    def _release() -> None:
        if released[0]:
            return
        released[0] = True
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            # Swallow KEPT -- release runs off the page's `destroyed`
            # signal and must never raise into Qt teardown -- but the
            # failure now leaves evidence instead of vanishing.
            _log.warning(
                "wizard audio-test stream stop/close failed during "
                "release", exc_info=True,
            )
        try:
            pa.terminate()
        except Exception:
            # Same contract: swallow KEPT, evidence logged (a failed
            # terminate leaks the PortAudio COM instance).
            _log.warning(
                "wizard audio-test PyAudio terminate failed during "
                "release", exc_info=True,
            )

    return _release


def _saved_audio_device() -> Optional[str]:
    """The merged config's ``audio.device`` when it names a real saved pick.

    Returns None for the shipped default (``"default"``), an empty/non-string
    value, or a config-read failure. Two consumers, both on
    :meth:`AudioPage.initializePage`:

    * the zero-device leg feeding :meth:`AudioPage.isComplete`: a machine
      that enumerates no loopback endpoints RIGHT NOW (e.g. an RDP session
      exposes no WASAPI loopback devices) but already carries a working
      saved device must not be hard-blocked -- the upgrade path FORCES the
      wizard (model pins changed), so a hard block strands a previously
      working install in the cancel -> exit-42 loop;
    * the devices-present leg, which preselects the saved device in the
      list so an untouched click-through round-trips it instead of
      writing the system default or the first row over it.

    Fails CLOSED (None) so a true first run, or a broken config, keeps
    today's behavior on both legs.
    """
    try:
        from yamabiko.config import load_full_config
        device = (load_full_config().get("audio") or {}).get("device")
    except Exception:
        _log.warning(
            "config read for audio.device failed; treating as no saved "
            "device", exc_info=True,
        )
        return None
    if isinstance(device, str) and device.strip() and device.strip() != "default":
        return device.strip()
    return None


def _default_list_devices() -> list[dict]:
    """Enumerate WASAPI loopback devices via PyAudioWPatch. Production default.

    Returns ``[{"index", "name"}]``; the loopback of the Windows default
    output also carries ``"default": True`` (the Sound step's "System
    default" row). The default is read from the SAME PyAudio instance:
    PortAudio indexes belong to the instance that listed them.
    """
    try:
        import pyaudiowpatch as pyaudio
        with pyaudio.PyAudio() as p:
            try:
                default_index = p.get_default_wasapi_loopback()["index"]
            except Exception:
                # No loopback analogue for the default output (LookupError)
                # or no WASAPI host API (OSError): list the devices unmarked.
                _log.info("no default WASAPI loopback; no System default mark", exc_info=True)
                default_index = None
            devs: list[dict] = []
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                # The PyAudioWPatch flag only, matching the capture-side
                # lookup (capture/audio.py): a microphone or virtual input
                # whose NAME merely contains "loopback" is not a WASAPI
                # loopback endpoint, and capture would never find it.
                if info.get("isLoopbackDevice"):
                    dev = {"index": i, "name": info.get("name", "")}
                    if i == default_index:
                        dev["default"] = True
                    devs.append(dev)
        return devs
    except Exception:
        # PyAudio()/PortAudio init raises when the Windows Audio service is
        # stopped/disabled or no endpoint exists. The wizard must not crash --
        # return [] so the page shows the no-device guidance (mirrors the
        # post-install Settings dialog's guard).
        _log.warning(
            "audio device enumeration failed; treating as no devices",
            exc_info=True,
        )
        return []


class AudioPage(QWizardPage):
    """The Sound step. Enumeration, preselect and the test keep today's
    logic; the value the wizard persists is the raw device name."""

    # Duration of the loopback test. Class attribute so tests can shrink it
    # instead of pumping a real 5 s event loop.
    _TEST_DURATION_MS: int = 5000
    # A test run whose session-max RMS never exceeds this floor is reported as
    # "no sound picked up" instead of success. Shares the runtime's single
    # silence floor (see the module import) so the wizard meter and the #54
    # idle heartbeat can't drift apart.
    _TEST_SILENCE_RMS_FLOOR: float = _SILENCE_RMS_FLOOR

    def __init__(
        self,
        *,
        list_devices: Optional[Callable[[], list[dict]]] = None,
        palette: dict = MIST,
    ) -> None:
        super().__init__()
        self.setTitle("Where does the sound play?")
        self.setSubTitle(
            "Pick the output your PC plays sound through. "
            "Yamabiko listens to it but never plays anything itself."
        )
        self._palette = palette
        self._list_devices = list_devices or _default_list_devices
        self._devices: list[dict] = []
        # Saved (non-default) audio.device from the merged config, resolved
        # in initializePage ONLY when zero devices enumerate. Lets a
        # previously configured install (e.g. over RDP, where no WASAPI
        # loopback endpoints exist) finish the forced upgrade wizard on its
        # saved device instead of hard-blocking -- see isComplete.
        self._saved_device: Optional[str] = None
        # A cancelled wizard stores its current device name in state.json.
        # Device enumeration is lazy, so the driver supplies that name here
        # and initializePage applies it only after the list is built.
        self._interim_device_name: Optional[str] = None

        # In-flight audio-test resources. Initialized here so
        # _release_test_resources is safe to call before any test ran.
        # stop_test releases a running test's PortAudio stream + instance
        # (another pick, entering or leaving the step, and the wizard's
        # release_background_threads when setup closes); the page's
        # `destroyed` signal releases too, but the setup window outlives
        # run_wizard, so that hook fires only at garbage collection.
        self._test_pa = None
        self._test_stream = None
        self._release_test_audio = None
        self._meter_timer = None
        self._finish_timer = None

        # Rebuilt by every enumeration (a SegmentedControl's options are
        # fixed); None while no device is listed.
        self.device_list: Optional[SegmentedControl] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 20, 0, 0)
        layout.setSpacing(0)

        # At most five rows show; more devices scroll inside the list only
        # (L8, the one exception to "nothing scrolls"). The scroll area owns
        # the closed frame (window QSS QScrollArea#deviceList): its edge
        # stays put and the scroll bar sits inside it while the rows scroll.
        self.device_area = QScrollArea()
        self.device_area.setObjectName("deviceList")
        self.device_area.setWidgetResizable(True)
        self.device_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.device_area.viewport().setAutoFillBackground(False)
        self.device_area.hide()
        layout.addWidget(self.device_area)

        self.no_device_alert = Alert(palette)
        self.look_again_button = self.no_device_alert.add_button("Look again")
        self.look_again_button.clicked.connect(lambda: self.initializePage())
        self.no_device_alert.hide()
        layout.addWidget(self.no_device_alert)

        layout.addSpacing(28)
        heading = QLabel("Test")
        heading.setFont(mincho_font(15))
        heading.setFixedHeight(20)
        layout.addWidget(heading)

        layout.addSpacing(10)
        test_row = QHBoxLayout()
        test_row.setSpacing(12)
        self.test_button = QPushButton("Test for 5 seconds")
        self.test_button.setAutoDefault(False)
        self.test_button.setEnabled(False)  # enabled once a device is listed
        self.test_button.clicked.connect(self._run_audio_test)
        hint = QLabel("Play something first.")
        hint.setObjectName("help")
        hint.setFont(ui_font(13))
        test_row.addWidget(self.test_button)
        test_row.addWidget(hint)
        test_row.addStretch(1)
        layout.addLayout(test_row)

        layout.addSpacing(14)
        self.meter = LevelMeter(palette)
        layout.addWidget(self.meter)

        layout.addSpacing(10)
        result_row = QHBoxLayout()
        result_row.setSpacing(8)
        self.result_mark = StateMark("done", palette)
        self.result_label = QLabel("")
        self.result_label.setFont(ui_font(14))
        result_row.addWidget(self.result_mark, 0, Qt.AlignmentFlag.AlignVCenter)
        result_row.addWidget(self.result_label)
        result_row.addStretch(1)
        layout.addLayout(result_row)
        self._show_result(None)
        layout.addStretch(1)

    def initializePage(self) -> None:  # noqa: N802
        # A rail jump back past this step cleans up only the page left (L2),
        # so a test started here can still run when Next enters the step
        # again: stop it before Test is enabled for a second run. A test cut
        # short has no result (cleanupPage's rule).
        if self.stop_test():
            self._show_result(None)
        # A choice made earlier in this session outranks the cancelled-
        # session seed on a Back/Next revisit (the list is rebuilt below).
        current_device_name = self.device_list.value() if self.device_list is not None else ""
        # Defensive: any list_devices implementation that raises (e.g. the
        # production enumerator on a stopped Windows Audio service) must not
        # propagate out of this Qt virtual and crash the whole wizard.
        try:
            self._devices = self._list_devices()
        except Exception:
            _log.warning(
                "audio device enumeration raised; treating as no devices",
                exc_info=True,
            )
            self._devices = []
        if not self._devices:
            # Zero devices enumerated: an install that already carries a
            # saved (non-default) audio.device may proceed on it -- the
            # list stays empty. A cancelled-session choice remains the
            # logical selection so the user can continue through a temporary
            # enumeration outage; a merged-config fallback still writes
            # nothing and therefore leaves its existing override untouched.
            if current_device_name:
                self._interim_device_name = current_device_name
            self._saved_device = self._interim_device_name or _saved_audio_device()
            self._show_devices(None)
            if self._saved_device is not None:
                self.no_device_alert.set_text(
                    DEVICES_GONE,
                    f"Yamabiko keeps using {_display_name(self._saved_device)}.",
                )
            else:
                self.no_device_alert.set_text(NO_DEVICES, NO_DEVICES_HELP)
            self.no_device_alert.show()
        else:
            # Devices (re)appeared on a revisit: drop the saved-device leg
            # and the zero-device alert.
            self._saved_device = None
            self.no_device_alert.hide()
            # Preselect: this session's pick > the cancelled session's >
            # the saved (non-default) audio.device > the system default >
            # the first row. accept() writes the CURRENT selection, and the
            # upgrade path forces completed installs back through this
            # wizard -- without the seed, an untouched click-through
            # overwrites a working saved pick (e.g. VB-Cable). Exact match
            # on the name accept() persists.
            self._show_devices(
                current_device_name or self._interim_device_name or _saved_audio_device()
            )
            # The list now owns the effective selection. Keeping the old
            # prior-session seed would overwrite a user's new choice when
            # initializePage runs again after Back/Next navigation.
            self._interim_device_name = None
        self.completeChanged.emit()

    def _show_devices(self, preferred: Optional[str]) -> None:
        """Rebuild the list from ``self._devices`` and select ``preferred``
        when it is listed, else the system default, else the first row."""
        old = self.device_list
        self.device_list = None
        if old is not None:
            self.device_area.takeWidget()
            old.deleteLater()
        self.test_button.setEnabled(bool(self._devices))
        if not self._devices:
            self.device_area.hide()
            return
        names = [dev["name"] for dev in self._devices]
        default = next((dev["name"] for dev in self._devices if dev.get("default")), None)
        value = preferred if preferred in names else (default or names[0])
        control = SegmentedControl(
            [
                SegmentOption(
                    dev["name"], _display_name(dev["name"]),
                    SYSTEM_DEFAULT if dev.get("default") else "",
                )
                for dev in self._devices
            ],
            self._palette, variant="list", wizard=True, framed=False, value=value,
        )
        control.setFont(ui_font(14))
        control.valueChanged.connect(self._on_device_changed)
        self.device_area.setWidget(control)
        # setWidget turns the list's background fill on (the palette's gray):
        # the unframed list shows the area's field tone instead.
        control.setAutoFillBackground(False)
        self.device_area.setFixedHeight(_list_height(min(len(self._devices), _MAX_ROWS)))
        self.device_area.show()
        self.device_list = control
        self._scroll_to_selection()

    def _scroll_to_selection(self) -> None:
        """Bring the selected row into the five visible ones (the system
        default can sit below the fifth row)."""
        if self.device_list is not None:
            self.device_area.ensureWidgetVisible(
                self.device_list.segment(self.device_list.value())
            )

    def showEvent(self, event) -> None:  # noqa: N802 -- Qt override
        super().showEvent(event)
        # Row geometry is final only after the step is laid out on screen.
        QTimer.singleShot(0, self, self._scroll_to_selection)

    def _on_device_changed(self, _value: str) -> None:
        # A result, and the level the meter holds with it, belongs to the
        # device it was measured on (Decisions 66, 140): a test still running
        # on the previous pick stops without a verdict. Arrow keys move the
        # choice without scrolling the list: bring a row below the fifth
        # into view (Decision 145).
        self.stop_test()
        self._show_result(None)
        self._scroll_to_selection()
        self.completeChanged.emit()

    def _show_result(self, state: Optional[str], text: str = "", level: float = 0.0) -> None:
        """The line under the meter (a state mark and its sentence) and the
        level the meter rests at: hidden and all off before the first test,
        while one runs (the meter timer drives the cells then) and after
        another pick; after a test that picked up sound the meter holds
        ``level``, the test's loudest, beside the check (Decision 140)."""
        self.meter.set_level(level)
        if state is None:
            self.result_mark.hide()
            self.result_label.hide()
            self.result_label.setText("")
            return
        self.result_mark.set_state(state)
        self.result_label.setText(text)
        self.result_mark.show()
        self.result_label.show()

    def isComplete(self) -> bool:  # noqa: N802
        if self.device_list is not None:
            return True
        # Zero enumerated devices: complete ONLY when a saved or cancelled-
        # session device exists (see initializePage) -- the forced upgrade
        # wizard must not brick a working install whose environment (RDP
        # session, headless boot) temporarily exposes no loopback endpoints.
        return self._saved_device is not None

    def selected_device_name(self) -> str:
        if self.device_list is not None:
            return self.device_list.value()
        return self._interim_device_name or ""

    def set_interim_device_name(self, name: str | None) -> None:
        """Seed a cancelled-session device after lazy enumeration."""
        self._interim_device_name = (
            name.strip() if isinstance(name, str) else None
        )

    def _run_audio_test(self) -> None:
        """Record 5 s with a callback-based stream and drive the meter.

        Uses PyAudio's callback mode (same as yamabiko/capture/audio.py)
        so the Qt main thread never blocks -- no processEvents() pump,
        no "Not Responding" from Windows.
        """
        import numpy as np
        import pyaudiowpatch as pyaudio

        device_name = self.selected_device_name().strip()
        if not device_name:
            return

        self.test_button.setEnabled(False)
        self._show_result(None)

        pa = None
        try:
            pa = pyaudio.PyAudio()
            # PortAudio indexes belong to the device list created by this
            # PyAudio instance. The list outlives the instance used to
            # populate it, so a cached index can be stale after an endpoint
            # failure or Windows device change. Resolve the persisted device
            # identity (its raw name) in the fresh list used for this Test
            # click instead of reusing an old numeric index.
            device_idx = None
            info = None
            for idx in range(pa.get_device_count()):
                candidate = pa.get_device_info_by_index(idx)
                if str(candidate.get("name", "")).strip() == device_name:
                    device_idx = idx
                    info = candidate
                    break
            if device_idx is None or info is None:
                raise OSError(
                    f"audio device is no longer available: {device_name}"
                )
            device_rate = int(info.get("defaultSampleRate") or 48000)
            device_channels = int(info.get("maxInputChannels") or 2)
            chunk = max(256, device_rate // 50)
        except Exception:
            _log.warning("wizard audio test could not open %r", device_name, exc_info=True)
            self._test_failed()
            # Release PortAudio on this failure path too, symmetric with the
            # pa.open() failure handling below. pa stays None when PyAudio()
            # itself raised (Windows Audio service stopped) -- nothing to
            # terminate then.
            if pa is not None:
                pa.terminate()
            return

        # Shared state. The callback writes _test_peak_rms on the audio
        # thread; the QTimer reads it + applies exponential-moving-average
        # smoothing on the main thread. Raw per-chunk RMS jitters wildly
        # on 20 ms windows of real audio, which makes the meter flicker.
        # EMA smooths the display without losing responsiveness.
        self._test_peak_rms: float = 0.0
        self._test_display_rms: float = 0.0
        # Session-wide max survives the per-tick peak-hold reset, so
        # _finish_test can tell "audio actually flowed" from "5 s of
        # silence" (wrong device picked / nothing playing).
        self._test_session_max_rms: float = 0.0
        # The loudest level the meter showed: what it holds after a test
        # that picked up sound (Decision 140).
        self._test_peak_level: float = 0.0
        self._test_pa = pa
        self._test_stream = None

        def _audio_callback(in_data, frame_count, time_info, status):
            pcm = np.frombuffer(in_data, dtype=np.float32)
            if device_channels > 1:
                pcm = pcm.reshape(-1, device_channels).mean(axis=1)
            if pcm.size:
                rms = float(np.sqrt(np.mean(pcm ** 2)))
                # Peak-hold: take the max since the last timer tick so brief
                # loud transients aren't missed between 100 ms polls.
                if rms > self._test_peak_rms:
                    self._test_peak_rms = rms
                if rms > self._test_session_max_rms:
                    self._test_session_max_rms = rms
            return (None, pyaudio.paContinue)

        try:
            self._test_stream = pa.open(
                format=pyaudio.paFloat32,
                channels=device_channels,
                rate=device_rate,
                input=True, input_device_index=device_idx,
                frames_per_buffer=chunk,
                stream_callback=_audio_callback,
            )
            self._test_stream.start_stream()
            # The per-run release: stop_test, the finish timer and the
            # page's `destroyed` signal share it. The closure holds the
            # handles directly and never touches the page (see
            # _make_test_audio_release); stale closures from earlier runs
            # are idempotent no-ops.
            release = _make_test_audio_release(self._test_stream, pa)
            self._release_test_audio = release
            self.destroyed.connect(release)
        except Exception:
            _log.warning("wizard audio test could not open %r", device_name, exc_info=True)
            self._test_failed()
            pa.terminate()
            # _test_pa was already assigned above; clear it so a later
            # release path can't terminate() the dead instance again.
            self._test_pa = None
            self._test_stream = None
            return

        # Poll the peak RMS every 100 ms, EMA-smoothed (alpha 0.35); reset
        # the peak-hold on each tick so the next interval captures its own
        # peak (not the session max).
        def _update_meter():
            alpha = 0.35
            peak = self._test_peak_rms
            self._test_peak_rms = 0.0
            self._test_display_rms = alpha * peak + (1 - alpha) * self._test_display_rms
            # The old 0-100 bar's scale (RMS x 400), as a fraction.
            level = min(1.0, self._test_display_rms * 4.0)
            self._test_peak_level = max(self._test_peak_level, level)
            self.meter.set_level(level)

        self._meter_timer = QTimer(self)
        self._meter_timer.setInterval(100)
        self._meter_timer.timeout.connect(_update_meter)
        self._meter_timer.start()

        def _finish_test():
            self._meter_timer.stop()
            self._release_test_resources()
            self.test_button.setEnabled(True)
            # Only claim success when audio actually flowed: a silent run
            # (wrong device picked, nothing playing) says so beside a dark
            # meter. A run that picked up sound keeps its loudest level lit,
            # at least one cell (Decision 140, screenshots 03/04).
            if self._test_session_max_rms > self._TEST_SILENCE_RMS_FLOOR:
                self._show_result(
                    "done", PICKING_UP,
                    level=max(self._test_peak_level, 1 / LevelMeter.CELLS),
                )
            else:
                self._show_result("paused", NO_SOUND)

        # Parented to the page, NOT a free-standing QTimer.singleShot: a page
        # destroyed inside the test window takes this timer with it, so
        # _finish_test can never run against the deleted C++ object.
        finish = QTimer(self)
        finish.setSingleShot(True)
        finish.timeout.connect(_finish_test)
        finish.start(self._TEST_DURATION_MS)
        self._finish_timer = finish

    def _test_failed(self) -> None:
        """The device couldn't be opened: say so beside a dark meter (the
        log has the error)."""
        self._show_result("paused", OPEN_FAILED)
        self.test_button.setEnabled(True)

    def _release_test_resources(self) -> None:
        """Stop + close the in-flight test stream and terminate the
        PyAudio instance, via the per-run idempotent closure. Whichever
        release path fires first wins -- _finish_test (normal expiry),
        stop_test (another pick, entering or leaving the step, setup
        closing), or the page's `destroyed` signal (garbage collection).
        """
        release, self._release_test_audio = self._release_test_audio, None
        self._test_stream = None
        self._test_pa = None
        if release is not None:
            release()

    def stop_test(self) -> bool:
        """Stop a running test: both timers and the open stream; Test is
        clickable again when a list exists. Returns whether a test was cut
        short (a finished test's result is the caller's to keep or clear)."""
        running = self._finish_timer is not None and self._finish_timer.isActive()
        if self._meter_timer is not None:
            self._meter_timer.stop()
        if self._finish_timer is not None:
            self._finish_timer.stop()
        self._release_test_resources()
        self.test_button.setEnabled(self.device_list is not None)
        return running

    def cleanupPage(self) -> None:  # noqa: N802 -- Qt convention
        # The user left the step (Back or a rail jump). The page OUTLIVES
        # that navigation, so the destroyed hook can't release here -- stop
        # the test now. A test cut short has no result, so its live meter
        # goes dark; a finished test keeps its result line and held meter
        # for a revisit (Decision 140).
        if self.stop_test():
            self._show_result(None)
        super().cleanupPage()
