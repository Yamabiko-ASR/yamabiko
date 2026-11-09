"""WASAPI loopback capture via PyAudioWPatch.

`sounddevice`/upstream PyAudio do not support WASAPI loopback on Windows.
PyAudioWPatch exposes loopback endpoints as duplicated virtual input devices.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyaudiowpatch as pyaudio

from yamabiko.state import PauseState

from ._audio_constants import (
    DEFAULT_IDLE_DEATH_S as _DEFAULT_IDLE_DEATH_S,
    SILENCE_RMS_FLOOR as _SILENCE_RMS_FLOOR,
    LoopbackDeviceNotFound,
)
from .default_endpoint import current_default_render_endpoint_id
from .envelope import (
    CaptureBlock,
    CaptureContinuityEvent,
    CaptureDiscontinuity,
    CaptureGapReason,
    GapExtent,
)
from .resampler import StreamingResampler

_log = logging.getLogger(__name__)

# Consecutive per-frame transform failures (with zero successful
# enqueues) after which the capture declares the stream dead. ~3 s of
# continuous failure at the 30 ms default block size — long enough to
# ride out a transient bad frame, short enough that a persistent
# buffer-size/channel mismatch (e.g. after a WASAPI session
# reconfigure) recovers via the watchdog instead of stalling forever.
_TRANSFORM_FAILURE_DEATH_THRESHOLD = 100


@dataclass
class AudioConfig:
    device: str = "default"
    sample_rate: int = 16000
    block_ms: int = 30
    idle_death_s: int = _DEFAULT_IDLE_DEATH_S


@dataclass(slots=True)
class _ContinuityRun:
    """Consecutive continuity events that share one INFO log key."""

    event_kind: str
    reason: CaptureGapReason | None
    stream_generation: int | None
    pause_epoch: int | None
    first_sequence: int
    last_sequence: int
    events: int = 0
    sample_count: int = 0

    def matches(self, event: CaptureContinuityEvent) -> bool:
        return (
            self.event_kind == event.event_kind
            and self.reason == event.reason
            and self.stream_generation == event.stream_generation
            and self.pause_epoch == event.pause_epoch
        )


def _log_continuity_run_ended(run: _ContinuityRun) -> None:
    _log.info(
        "capture continuity run ended event_kind=%s reason=%s "
        "stream_generation=%s pause_epoch=%s events=%d sample_count=%d "
        "first_sequence=%d last_sequence=%d",
        run.event_kind,
        run.reason.value if run.reason is not None else None,
        run.stream_generation,
        run.pause_epoch,
        run.events,
        run.sample_count,
        run.first_sequence,
        run.last_sequence,
    )


class LoopbackCapture:
    """Capture default-output loopback at 16 kHz mono into an asyncio.Queue.

    The PortAudio callback runs on its own OS thread; bridge it into the
    asyncio loop with `loop.call_soon_threadsafe` to avoid queue-type mixing.
    """

    def __init__(
        self,
        config: AudioConfig,
        loop: asyncio.AbstractEventLoop,
        out_queue: asyncio.Queue,
        *,
        on_stream_death=None,
        on_capture_event: (
            Callable[[CaptureContinuityEvent], None] | None
        ) = None,
        pause_state_probe: Callable[[], PauseState] | None = None,
    ):
        self.config = config
        self._loop = loop
        self._out = out_queue
        self._pa: pyaudio.PyAudio | None = None
        self._stream = None
        self._device_info: dict | None = None
        # When capturing the OS default output (config.device ==
        # "default"), we snapshot the default-render endpoint ID at start()
        # so the monitor can notice a runtime switch. The old loopback keeps
        # is_active()==True after a switch but yields only silence, so the
        # is_active() probe can never catch it; subtitles die with no error
        # (README told users to "restart"). The ID comes from Windows Core
        # Audio (default_endpoint.current_default_render_endpoint_id) -- NOT
        # PyAudio, whose default is frozen at Pa_Initialize() and never
        # updates (pyaudiowpatch has no Pa_RefreshDeviceList). Stays None for
        # a pinned named device -- their explicit choice is independent of the
        # system default, so we must NOT bounce them.
        self._default_output_id: str | None = None
        self._stop = threading.Event()
        # Callback invoked (once) when the capture stream dies unexpectedly
        # -- device unplug, WASAPI session reset. Settable after construction
        # (run_pipeline wires it to signal the pipeline's shutdown). Defaults to
        # None; death is always logged regardless.
        self.on_stream_death = on_stream_death
        self.on_capture_event = on_capture_event
        self._pause_state_probe = pause_state_probe or (
            lambda: PauseState(epoch=0, paused=False)
        )
        self._monitor: threading.Thread | None = None
        self._death_signalled = False
        # _signal_death is reachable from TWO foreign threads (the
        # stream-death monitor and PortAudio's callback thread). The lock
        # makes the claimed-once flag atomic; without it both threads
        # could pass the guard (benign: double log + idempotent callback,
        # but the once-contract should hold by construction).
        self._death_lock = threading.Lock()
        # Serializes teardown of the PortAudio stream + instance between the
        # monitor's in-place re-open (#54) and stop(): both mutate
        # self._stream / self._pa, and without this they could double-close /
        # use-after-free the C handles.
        self._stream_lock = threading.Lock()
        # The reason passed to the winning _signal_death() call,
        # captured so consumers (and tests) can see WHY the stream was
        # declared dead -- default-output switch vs unplug vs transform.
        self._last_death_reason: str | None = None
        # Count frames dropped under back-pressure for throttled logging.
        self._dropped = 0
        # A full queue cannot accept a control marker immediately. Once any
        # capture time is lost, drop later PCM until every ordered notice has
        # entered the queue behind all already-buffered audio. The callback
        # lock only accounts for rare notices; marker delivery and telemetry
        # stay on the asyncio loop.
        self._discontinuity_pending = threading.Event()
        self._discontinuity_notice_lock = threading.Lock()
        self._unregistered_discontinuity_notices = 0
        self._discontinuity_notices: deque[CaptureDiscontinuity] = deque()
        self._discontinuity_marker_in_flight: CaptureDiscontinuity | None = None
        self._discontinuity_task: asyncio.Task | None = None
        self._capture_event_sequence = 0
        self._next_stream_generation = 0
        # Continuity telemetry is logged at INFO once per run of events that
        # share kind, reason, generation and pause epoch. The run is touched on
        # the loop thread and by stop(), which may run on a worker thread; the
        # lock guards only those two, never the PortAudio callback.
        self._continuity_run: _ContinuityRun | None = None
        self._continuity_lock = threading.Lock()
        # Transform-failure accounting. Total drives the log
        # throttle (1st + every 100th gets a full traceback);
        # consecutive drives the death escalation and resets whenever
        # a frame makes it through the transform (a QueueFull drop in
        # _enqueue still resets — back-pressure is not a transform
        # failure).
        self._transform_failures = 0
        self._consecutive_transform_failures = 0
        # Idle-audio re-open (#54). The realtime callback stamps these (atomic
        # scalar writes under the GIL -- no lock, same as _dropped above); the
        # monitor reads them in _check_idle_reopen. _last_audible + _ever_audible
        # advance only on ABOVE-FLOOR energy. On WASAPI loopback a silent
        # endpoint delivers NO callbacks at all, so a dead endpoint and a
        # paused/idle one are indistinguishable by frame arrival -- which is why
        # the heartbeat's action is a cheap in-place STREAM RE-OPEN (harmless on
        # a benign pause; re-attaches a fresh stream so audio that RESUMES on
        # the pinned endpoint is captured again), not a process restart, and why
        # it is gated on _ever_audible (a stream that never produced audio is
        # idle/not-yet-playing, not stalled). _last_frame advances on every
        # enqueue but production decides on audible energy, so it is read only
        # by a real-tier liveness test. GIL-atomicity assumed;
        # revisit under a free-threaded build (do NOT lock the realtime thread).
        self._last_audible_monotonic: float | None = None
        self._last_frame_monotonic: float | None = None
        self._ever_audible: bool = False
        self._silence_floor = _SILENCE_RMS_FLOOR
        self._idle_death_s = config.idle_death_s
        # Live pause state, read through pause_state_probe (build_pipeline
        # passes the app state's snapshot); the default never pauses. Avoids
        # re-open churn while Yamabiko itself is paused (its own pause, not
        # the source player's -- the re-open is harmless either way, so this
        # is only a churn guard).
        self._is_paused = lambda: self._pause_state_probe().paused
        # Monitor poll interval. A test seam: shrinking it lets a test cross
        # the idle window in well under a second instead of the 90 s default.
        self._monitor_poll_s: float = 1.0
        # Device-resolution grace (#54). On a re-open / restart that races a
        # reconnecting device (unplug/replug, or a suspended endpoint settling),
        # retry resolution a few times before surfacing the no-audio dialog.
        # ~3 x 2 s ~= 6 s. Bounded so a genuinely-absent device still surfaces
        # promptly; interruptible via _stop. Test seams.
        self._device_grace_attempts: int = 3
        self._device_grace_poll_s: float = 2.0

    def _resolve_device_with_grace(self) -> dict:
        """Resolve the configured device, retrying briefly if it isn't present
        on the first attempt.

        A capture re-open / restart can race a device that is mid-reconnect --
        the user unplugged/replugged, or a suspended endpoint is still settling.
        A short bounded grace lets it reappear instead of failing straight to
        the no-audio dialog. pyaudiowpatch freezes its device list at PyAudio()
        init (no Pa_RefreshDeviceList), so each retry re-creates the instance to
        see a newly-present device. Bounded so a genuinely-absent device still
        surfaces the dialog promptly, and interruptible via _stop so a shutdown
        mid-grace aborts at once.
        """
        try:
            return self._resolve_device()
        except RuntimeError as exc:
            last_exc = exc
        for attempt in range(1, self._device_grace_attempts + 1):
            # Interruptible: _stop.wait returns True if stop() fired -> abort.
            if self._stop.wait(self._device_grace_poll_s):
                break
            try:
                if self._pa is not None:
                    self._pa.terminate()
            except Exception:
                pass
            self._pa = pyaudio.PyAudio()
            try:
                info = self._resolve_device()
            except RuntimeError as exc:
                last_exc = exc
                continue
            _log.info(
                "audio device resolved on grace retry %d/%d",
                attempt, self._device_grace_attempts,
            )
            return info
        raise last_exc

    def _resolve_device(self) -> dict:
        pa = self._pa
        assert pa is not None
        if self.config.device == "default":
            try:
                return pa.get_default_wasapi_loopback()
            except Exception:
                default_out = pa.get_default_output_device_info()
                for i in range(pa.get_device_count()):
                    info = pa.get_device_info_by_index(i)
                    if info.get("isLoopbackDevice") and default_out["name"] in info["name"]:
                        return info
                raise RuntimeError("No WASAPI loopback device found for default output")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("isLoopbackDevice") and self.config.device in info["name"]:
                return info
        raise LoopbackDeviceNotFound(self.config.device)

    def _enqueue(self, block: CaptureBlock) -> None:
        """Put a frame on the out-queue from the loop thread. Catch QueueFull
        so back-pressure (ASR behind) drops the frame with a throttled warning
        instead of raising an uncaught asyncio.QueueFull on the event loop
        every callback (silent frame loss + per-frame spam)."""
        if self._discontinuity_pending.is_set():
            self._record_dropped_block(block, reason=None)
            return
        try:
            self._out.put_nowait(block)
        except asyncio.QueueFull:
            self._record_dropped_block(
                block,
                reason=CaptureGapReason.AUDIO_QUEUE_OVERFLOW,
            )
            self._request_discontinuity(CaptureDiscontinuity(
                reason=CaptureGapReason.AUDIO_QUEUE_OVERFLOW,
                extent=GapExtent.EXACT_OUTPUT_SAMPLES,
                stream_generation=block.stream_generation,
                at_sample=block.start_sample,
            ))
        else:
            self._flush_continuity_run()

    def _record_dropped_frame(self) -> None:
        self._dropped += 1
        if self._dropped == 1 or self._dropped % 100 == 0:
            _log.warning(
                "audio out-queue full or discontinuous; dropping frames "
                "until the reset marker is delivered -- total dropped: %d",
                self._dropped,
            )

    def _record_dropped_block(
        self,
        block: CaptureBlock,
        *,
        reason: CaptureGapReason | None,
    ) -> None:
        self._record_dropped_frame()
        self._emit_capture_event(
            event_kind="output_dropped",
            extent=GapExtent.EXACT_OUTPUT_SAMPLES,
            reason=reason,
            stream_generation=block.stream_generation,
            start_sample=block.start_sample,
            sample_count=int(block.pcm.size),
            pause_epoch=block.pause_epoch,
        )

    def _record_paused_block(self, block: CaptureBlock) -> None:
        """Record one exactly measured converted block intentionally muted."""

        self._emit_capture_event(
            event_kind="paused_discard",
            extent=GapExtent.EXACT_OUTPUT_SAMPLES,
            reason=CaptureGapReason.USER_PAUSE,
            stream_generation=block.stream_generation,
            start_sample=block.start_sample,
            sample_count=int(block.pcm.size),
            pause_epoch=block.pause_epoch,
        )

    def _record_suspected_source_gap(
        self,
        stream_generation: int,
        start_sample: int,
        sample_count: int,
        pause_epoch: int,
    ) -> None:
        """Record a WASAPI input-clock jump without touching the PCM path."""

        self._emit_capture_event(
            event_kind="suspected_source_gap",
            extent=GapExtent.UNKNOWN_SOURCE_DURATION,
            reason=CaptureGapReason.INPUT_STATUS_GAP,
            stream_generation=stream_generation,
            start_sample=start_sample,
            sample_count=sample_count,
            pause_epoch=pause_epoch,
        )

    def _emit_capture_event(
        self,
        *,
        event_kind: str,
        extent: GapExtent | None,
        reason: CaptureGapReason | None,
        stream_generation: int | None,
        start_sample: int | None,
        sample_count: int | None,
        pause_epoch: int | None = None,
    ) -> None:
        """Publish content-free continuity telemetry from the loop thread.

        INFO marks where a run of same-key events starts and ends; each event
        is logged at DEBUG, and the sink still receives every event.
        """
        self._capture_event_sequence += 1
        event = CaptureContinuityEvent(
            event_sequence=self._capture_event_sequence,
            event_kind=event_kind,
            extent=extent,
            reason=reason,
            stream_generation=stream_generation,
            start_sample=start_sample,
            sample_count=sample_count,
            pause_epoch=pause_epoch,
        )
        reason_value = event.reason.value if event.reason is not None else None
        with self._continuity_lock:
            run = self._continuity_run
            if run is None or not run.matches(event):
                if run is not None:
                    _log_continuity_run_ended(run)
                _log.info(
                    "capture continuity run started event_kind=%s reason=%s "
                    "stream_generation=%s pause_epoch=%s start_sample=%s",
                    event.event_kind,
                    reason_value,
                    event.stream_generation,
                    event.pause_epoch,
                    event.start_sample,
                )
                run = self._continuity_run = _ContinuityRun(
                    event_kind=event.event_kind,
                    reason=event.reason,
                    stream_generation=event.stream_generation,
                    pause_epoch=event.pause_epoch,
                    first_sequence=event.event_sequence,
                    last_sequence=event.event_sequence,
                )
            run.last_sequence = event.event_sequence
            run.events += 1
            run.sample_count += event.sample_count or 0
        _log.debug(
            "capture continuity event_sequence=%d event_kind=%s extent=%s "
            "reason=%s stream_generation=%s start_sample=%s sample_count=%s "
            "pause_epoch=%s",
            event.event_sequence,
            event.event_kind,
            event.extent.value if event.extent is not None else None,
            reason_value,
            event.stream_generation,
            event.start_sample,
            event.sample_count,
            event.pause_epoch,
        )
        if self.on_capture_event is not None:
            try:
                self.on_capture_event(event)
            except Exception:
                # A sink is observational and must never break audio delivery.
                # Do not include its exception text in privacy-safe telemetry.
                _log.error("capture continuity event sink failed")

    def _flush_continuity_run(self) -> None:
        """Log the open continuity run's summary and close it.

        Only the loop thread opens runs, so an unlocked None read is safe. The
        swap is locked because stop() may flush from a worker thread.
        """
        if self._continuity_run is None:
            return
        with self._continuity_lock:
            run, self._continuity_run = self._continuity_run, None
            if run is not None:
                _log_continuity_run_ended(run)

    def _request_discontinuity(
        self,
        reason: CaptureDiscontinuity | CaptureGapReason,
    ) -> None:
        """Arrange an ordered marker without ever blocking capture threads."""
        notice = (
            reason
            if isinstance(reason, CaptureDiscontinuity)
            else CaptureDiscontinuity(reason=reason)
        )
        with self._discontinuity_notice_lock:
            self._unregistered_discontinuity_notices += 1
            self._discontinuity_pending.set()
            # Hold the rare-path lock through scheduling so two producer
            # threads cannot invert the order established by lock acquisition.
            self._loop.call_soon_threadsafe(
                self._accept_discontinuity_notice,
                notice,
            )

    def apply_pause_edge(self, edge: PauseState) -> None:
        """Order one pause control behind queued PCM without blocking capture."""

        if not isinstance(edge, PauseState):
            raise TypeError("edge must be a PauseState")
        self._flush_continuity_run()
        self._request_discontinuity(CaptureDiscontinuity(
            reason=(
                CaptureGapReason.USER_PAUSE
                if edge.paused
                else CaptureGapReason.USER_RESUME
            ),
            pause_epoch=edge.epoch,
        ))

    def _accept_discontinuity_notice(
        self,
        notice: CaptureDiscontinuity,
    ) -> None:
        """Register one notice on the loop in observed callback order."""
        with self._discontinuity_notice_lock:
            self._unregistered_discontinuity_notices -= 1
        if notice.extent is GapExtent.UNKNOWN_SOURCE_DURATION:
            self._emit_capture_event(
                event_kind="source_gap",
                extent=notice.extent,
                reason=notice.reason,
                stream_generation=notice.stream_generation,
                start_sample=notice.at_sample,
                sample_count=None,
                pause_epoch=notice.pause_epoch,
            )
        latest = (
            self._discontinuity_notices[-1]
            if self._discontinuity_notices
            else self._discontinuity_marker_in_flight
        )
        # Repeated equivalent reset markers add no downstream information.
        # Their individual source-gap observations were still emitted above.
        if (
            latest is None
            or latest.reason is not notice.reason
            or latest.extent is not notice.extent
        ):
            self._discontinuity_notices.append(notice)
        self._ensure_discontinuity_delivery()

    def _ensure_discontinuity_delivery(self) -> None:
        task = self._discontinuity_task
        if task is not None and not task.done():
            return
        self._discontinuity_task = asyncio.create_task(
            self._deliver_discontinuities(),
        )

    async def _deliver_discontinuities(self) -> None:
        while True:
            if self._discontinuity_notices:
                notice = self._discontinuity_notices.popleft()
                self._discontinuity_marker_in_flight = notice
                try:
                    await self._out.put(notice)
                finally:
                    self._discontinuity_marker_in_flight = None
                continue
            with self._discontinuity_notice_lock:
                if self._unregistered_discontinuity_notices == 0:
                    self._discontinuity_pending.clear()
                    return
            # A foreign thread has scheduled registration but its loop
            # callback has not run yet. Yield without collapsing the notice.
            await asyncio.sleep(0)

    def _read_default_output_id(self) -> str | None:
        """The CURRENT OS default-render endpoint ID, or None if it can't
        be read right now.

        Snapshotted at start() and polled once a second by the monitor.
        Delegates to Windows Core Audio (current_default_render_endpoint_id),
        which always reflects the live default -- unlike PyAudio, whose default
        is frozen at Pa_Initialize(). None means "can't tell" (no output
        device, or a transient COM failure mid-switch); the caller treats it
        as no-change rather than a switch, so a flaky poll never bounces the
        stream. Kept as an instance method purely as a test seam.
        """
        return current_default_render_endpoint_id()

    def endpoint_name(self) -> str | None:
        """The captured endpoint's Windows friendly name, as Core Audio
        lists it (PyAudio's loopback name without " [Loopback]"), or None
        before start(). The pipeline names it in the audio problem."""
        info = self._device_info
        name = info.get("name") if isinstance(info, dict) else None
        if not isinstance(name, str) or not name:
            return None
        return name.removesuffix(" [Loopback]")

    def start(self) -> None:
        self._pa = pyaudio.PyAudio()
        try:
            self._device_info = self._resolve_device_with_grace()
        except LoopbackDeviceNotFound as exc:
            if self._stop.is_set():
                raise
            # ``AudioConfig`` is a per-process copy built from the loaded
            # config. Switching this copy does not overwrite the user's saved
            # preference, so a reconnected headset is selected next launch.
            # For this session, follow the Windows default instead of turning
            # an ordinary unplugged-device state into a terminal startup error.
            self.config.device = "default"
            _log.warning(
                "configured audio device %r is unavailable; using the "
                "Windows default output for this session",
                exc.device_name,
            )
            self._device_info = self._resolve_device()
        # Only track the default-output endpoint when we're actually
        # following the default; a pinned named device is independent of it.
        if self.config.device == "default":
            self._default_output_id = self._read_default_output_id()
        self._open_stream()
        self._start_monitor()

    def _open_stream(self) -> None:
        """Open + start the PortAudio loopback stream against the resolved
        self._device_info and (re-)seed the idle window. Shared by start() and
        the in-place re-open. Assumes self._pa and self._device_info are set.
        """
        info = self._device_info
        assert self._pa is not None and info is not None
        device_rate = int(info["defaultSampleRate"])
        device_channels = int(info["maxInputChannels"])
        frames_per_buffer = int(device_rate * self.config.block_ms / 1000)
        stream_generation = self._next_stream_generation
        self._next_stream_generation += 1
        next_output_sample = 0
        # One filter/phase state belongs to exactly one PortAudio stream
        # generation. Re-open constructs a fresh callback and converter; stop
        # and re-open deliberately discard the old tail without flushing it
        # beside audio from a different timeline.
        converter = (
            StreamingResampler(device_rate, self.config.sample_rate)
            if device_rate != self.config.sample_rate
            else None
        )
        last_pause_epoch = self._pause_state_probe().epoch
        # Input ADC time of the previous callback in this stream; 0.0 means
        # unknown. Local to this open, so a re-open starts with no baseline.
        prev_adc = 0.0

        def cb(in_data, frame_count, time_info, status):
            nonlocal last_pause_epoch, next_output_sample, prev_adc
            if self._stop.is_set():
                return (None, pyaudio.paComplete)
            # This callback runs on PortAudio's own thread. An unhandled
            # exception here kills the stream silently -- the main loop
            # keeps running, but no further frames arrive. Catch everything,
            # log (throttled), and return paContinue so a single bad frame
            # doesn't take down the whole capture. PERSISTENT failure escalates
            # to stream death after the consecutive threshold.
            try:
                pause_snapshot = self._pause_state_probe()
                if pause_snapshot.epoch != last_pause_epoch:
                    if converter is not None:
                        converter.reset()
                    last_pause_epoch = pause_snapshot.epoch
                # A jump in the input ADC clock beyond three callback
                # durations means source time went missing although no status
                # flag says so. Telemetry only: no marker, no converter reset
                # and no dropped PCM. Checked before decoding so a failed
                # transform does not skew the next comparison.
                adc = (
                    time_info.get("input_buffer_adc_time") if time_info else None
                ) or 0.0
                frame_seconds = frame_count / device_rate
                if prev_adc > 0 and adc > 0 and adc - prev_adc > 3 * frame_seconds:
                    self._loop.call_soon_threadsafe(
                        self._record_suspected_source_gap,
                        stream_generation,
                        next_output_sample,
                        round(
                            (adc - prev_adc - frame_seconds)
                            * self.config.sample_rate,
                        ),
                        pause_snapshot.epoch,
                    )
                prev_adc = adc
                pcm = np.frombuffer(in_data, dtype=np.float32).reshape(-1, device_channels)
                if device_channels > 1:
                    pcm = pcm.mean(axis=1)
                else:
                    pcm = pcm[:, 0]
                pcm = np.ascontiguousarray(pcm, dtype=np.float32)
                if not np.isfinite(pcm).all():
                    raise ValueError("WASAPI callback contains non-finite PCM")

                # PortAudio's WASAPI host API never sets the underflow/overflow
                # flags, so this branch serves other host APIs; WASAPI timing
                # gaps surface as suspected_source_gap above.
                source_gap = bool(status and status & (
                    pyaudio.paInputUnderflow | pyaudio.paInputOverflow
                ))

                # Idle-reopen liveness (#54): frame stamp advances on every
                # valid input callback even when SoXR retains all output for a
                # later batch (read only by a real-tier test); the audible
                # stamp + sticky _ever_audible advance only when input RMS
                # clears the floor (non-finite frames were rejected above).
                # _ever_audible gates the idle
                # re-open: a stream that never produced audible audio is
                # idle/not-yet-playing, not stalled.
                now = time.monotonic()
                self._last_frame_monotonic = now
                if pcm.size:
                    rms = float(
                        np.sqrt(np.mean(np.square(pcm, dtype=np.float64))),
                    )
                    if rms > self._silence_floor:
                        self._last_audible_monotonic = now
                        self._ever_audible = True

                if source_gap:
                    # PortAudio reports missing source time. Discard the old
                    # filter state without flushing. Queue back-pressure still
                    # happens after continuous resampling; the ordered marker
                    # resets downstream VAD for any missing output.
                    self._request_discontinuity(CaptureDiscontinuity(
                        reason=CaptureGapReason.INPUT_STATUS_GAP,
                        stream_generation=stream_generation,
                        at_sample=next_output_sample,
                    ))
                    if converter is not None:
                        converter.reset()

                output = converter.process(pcm) if converter is not None else pcm
                # SoXR emits variable-length batches, including empty arrays
                # while filter delay accumulates. The Silero accumulator accepts
                # arbitrary non-empty blocks; do not invent fixed callback
                # counts or enqueue empty work.
                if output.size:
                    block = CaptureBlock(
                        output.copy(),
                        stream_generation,
                        next_output_sample,
                        pause_epoch=pause_snapshot.epoch,
                    )
                    next_output_sample += int(output.size)
                    self._loop.call_soon_threadsafe(
                        (
                            self._record_paused_block
                            if pause_snapshot.paused
                            else self._enqueue
                        ),
                        block,
                    )
                self._consecutive_transform_failures = 0
            except Exception:
                if converter is not None:
                    converter.reset()
                self._request_discontinuity(CaptureDiscontinuity(
                    reason=CaptureGapReason.TRANSFORM_FAILURE,
                    stream_generation=stream_generation,
                    at_sample=next_output_sample,
                ))
                if self._record_transform_failure():
                    return (None, pyaudio.paComplete)
            return (None, pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=device_channels,
            rate=device_rate,
            input=True,
            input_device_index=info["index"],
            frames_per_buffer=frames_per_buffer,
            stream_callback=cb,
        )
        self._stream.start_stream()
        # Seed the idle window at stream-open and clear the audible-proof flag:
        # a brief startup silence is inside the window, the first monitor tick
        # never reads a None timestamp, and a FRESH stream hasn't proven audio
        # yet -- so the idle re-open won't fire until audio plays and then stops.
        now = time.monotonic()
        self._last_audible_monotonic = self._last_frame_monotonic = now
        self._ever_audible = False

    def _start_monitor(self) -> None:
        """Detect stream death (unplug / WASAPI reset / default-output switch)
        and re-open the stream in place when a live endpoint goes idle.

        PortAudio reports a hard-dead stream via ``is_active() == False``. The
        default-output-switch poll catches a runtime default switch that leaves
        the old loopback is_active()==True but silent. The idle-reopen check
        (#54) re-attaches a live endpoint that went silent. We poll once a
        second; a genuine death invokes ``on_stream_death`` (run_pipeline wires
        it to a graceful shutdown so the watchdog can restart + re-resolve).
        """
        def _watch() -> None:
            # self._stop.wait returns True when stop() is called -> exit quietly
            # (a deliberate stop is not a death).
            while not self._stop.wait(timeout=self._monitor_poll_s):
                stream = self._stream
                if stream is None:
                    continue  # mid re-open / between streams -> try next tick
                try:
                    active = stream.is_active()
                except Exception:
                    active = False  # querying a torn-down stream -> treat as dead
                # Re-check the stop flag AFTER the is_active probe.
                # stop() sets _stop first, then stop_stream() (which
                # makes is_active() False), then clears self._stream —
                # so a tick landing inside that window sees a non-None,
                # inactive stream during a DELIBERATE stop. Without the
                # re-check, that race logged the device-unplugged ERROR
                # and fired on_stream_death on every clean shutdown the
                # monitor happened to observe mid-teardown.
                if not active and not self._stop.is_set():
                    self._signal_death()
                    return

                # A runtime switch of the OS DEFAULT OUTPUT device
                # leaves this loopback is_active()==True but silent, so the
                # is_active() probe above can never catch it. Only relevant
                # when we're following the default (a pinned named device is
                # independent of the system default) and we managed to
                # snapshot a baseline at start(). _read_default_output_id
                # returns None on a transient failure mid-switch -- treated as
                # "can't tell, try next tick", never as a change. Same
                # stop-flag re-check before firing.
                if (
                    self.config.device == "default"
                    and self._default_output_id is not None
                ):
                    current = self._read_default_output_id()
                    if (
                        current is not None
                        and current != self._default_output_id
                        and not self._stop.is_set()
                    ):
                        self._signal_death(
                            reason="default output device changed",
                        )
                        return

                # Idle-audio re-open (#54): an endpoint that produced audio then
                # went silent for the window (suspended / render session moved)
                # is re-attached by re-opening the capture stream in place. On
                # WASAPI loopback a benign pause looks identical (no callbacks
                # during silence), so the action is a cheap re-open -- harmless
                # on a pause, re-attaching so resuming audio is caught -- NOT a
                # process restart. If the device can no longer be resolved
                # (genuinely gone), escalate to a full restart to re-resolve.
                reason = self._check_idle_reopen(time.monotonic())
                if reason is not None and not self._stop.is_set():
                    try:
                        self._reopen_stream()
                    except Exception:
                        if not self._stop.is_set():
                            self._signal_death(
                                reason=f"{reason}; in-place re-open failed (device gone)",
                            )
                        return
                    _log.warning(
                        "capture idle (%s); re-opened the stream in place", reason,
                    )

        self._monitor = threading.Thread(
            target=_watch, name="audio-stream-monitor", daemon=True,
        )
        self._monitor.start()

    def _record_transform_failure(self) -> bool:
        """Account one failed per-frame transform; return True when
        the stream should stop because the transform is persistently dead.

        Runs on PortAudio's realtime thread, inside the callback's
        except block (so ``_log.exception`` has exc_info). The full
        traceback fires on the 1st and every 100th total failure only
        (throttled) -- the unthrottled version pushed a multi-line
        traceback through the rotating file handler ~33x/s from the
        realtime thread. After ``_TRANSFORM_FAILURE_DEATH_THRESHOLD``
        consecutive failures with zero successful enqueues, signal death:
        the stream still reports ``is_active() == True`` (the callback
        keeps returning paContinue), so the stream-death monitor can
        never catch this state and the pipeline would otherwise stall
        with no audio forever.
        """
        self._transform_failures += 1
        self._consecutive_transform_failures += 1
        if self._transform_failures == 1 or self._transform_failures % 100 == 0:
            _log.exception(
                "WASAPI callback transform failed; dropping frame "
                "(total transform failures: %d)", self._transform_failures,
            )
        if (
            self._consecutive_transform_failures
            >= _TRANSFORM_FAILURE_DEATH_THRESHOLD
        ):
            self._signal_death(
                reason=(
                    "per-frame transform failing persistently "
                    f"({self._consecutive_transform_failures} consecutive "
                    "frames, e.g. buffer/channel mismatch after a WASAPI "
                    "session reconfigure)"
                ),
            )
            return True
        return False

    def _check_idle_reopen(self, now: float) -> str | None:
        """Return a reason if the stream has gone idle (no audible audio for the
        window after audio WAS flowing) and should be re-opened in place, else
        None. Pure given (now, the audible stamp, _ever_audible, pause state),
        so the monitor stays a thin driver and this is directly unit-testable.

        Gated on _ever_audible: a stream that has NEVER produced above-floor
        audio since it opened is idle / not-yet-playing, not stalled -- firing
        there would churn on cold start (app launched before playback) and on a
        source that's paused at launch. Only a stream that produced audio and
        then went silent for the window is a candidate.

        The discriminator is audible ENERGY, not frame arrival: on WASAPI
        loopback a silent endpoint delivers NO callbacks at all, so a paused
        source and a dead one look identical here. That ambiguity is fine
        because the ACTION (see _reopen_stream) is a cheap in-place re-open,
        harmless on a benign pause and re-attaching a fresh stream so audio that
        resumes on the pinned endpoint is captured. A resolvable endpoint that
        stays permanently silent (audio moved elsewhere for good) is NOT
        auto-recoverable by re-open or restart -- both re-attach the same silent
        endpoint; that narrow tail is a conscious soft-launch limitation.
        """
        audible = self._last_audible_monotonic
        if audible is None or not self._ever_audible:
            return None  # never produced audio -> idle / not-yet-playing
        if self._is_paused():
            # Yamabiko's own pause: reset the window so resume starts fresh.
            self._last_audible_monotonic = now
            return None
        if now - audible < self._idle_death_s:
            return None  # audible audio within the window -> alive
        return (
            f"no audible audio for {now - audible:.0f}s after audio was flowing "
            f"(endpoint may have suspended or its render session moved)"
        )

    def _reopen_stream(self) -> None:
        """Re-establish the PortAudio capture stream in place -- re-resolve the
        device, fresh PyAudio + stream against the SAME out-queue -- without
        tearing down the pipeline.

        Triggered by the idle check: a fresh stream re-attaches to the
        endpoint's current render session, so audio that resumed or moved back
        is captured again. Harmless if the silence was a benign pause -- the
        fresh stream simply waits for frames, exactly as the old one would have,
        and (because _ever_audible is reset on open) it won't re-fire until
        audio plays and stops again, so a silence episode triggers exactly one
        re-open, not a loop.

        Raises if the device can no longer be resolved (genuinely gone) -- the
        monitor escalates that to _signal_death (a full restart that
        re-resolves). Serialized with stop() via _stream_lock.
        """
        with self._stream_lock:
            if self._stop.is_set():
                return  # shutting down -> don't rebuild
            old_stream, self._stream = self._stream, None
            if old_stream is not None:
                try:
                    old_stream.stop_stream()
                except Exception:
                    pass
                try:
                    old_stream.close()
                except Exception:
                    pass
            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None
            self._pa = pyaudio.PyAudio()
            self._device_info = self._resolve_device_with_grace()
            # _open_stream below assigns this generation to the new stream,
            # whose output clock starts at sample 0.
            self._request_discontinuity(CaptureDiscontinuity(
                reason=CaptureGapReason.STREAM_REOPENED,
                stream_generation=self._next_stream_generation,
                at_sample=0,
            ))
            self._open_stream()

    def _signal_death(self, reason: str | None = None) -> None:
        # Claim the once-flag atomically; the log + callback run outside
        # the lock so a slow on_stream_death can't block the other
        # signalling thread.
        with self._death_lock:
            if self._death_signalled:
                return
            self._death_signalled = True
            # Record the winning reason inside the lock, before the
            # callback runs, so on_stream_death consumers can read it.
            self._last_death_reason = reason
        _log.error(
            "audio capture stream is dead: %s; no further frames will arrive",
            reason or (
                "stream no longer active (device unplugged or WASAPI "
                "session reset)"
            ),
        )
        if self.on_stream_death is not None:
            try:
                self.on_stream_death()
            except Exception:
                _log.exception("on_stream_death callback raised")

    def stop(self) -> None:
        # Tear down every resource even if an earlier step raises, and
        # never let stop() propagate -- it runs inside run_pipeline's finally,
        # where a raise would skip the remaining cleanup. Each step is guarded.
        self._stop.set()  # also unblocks + ends the monitor thread + grace waits
        with self._stream_lock:  # serialize with the monitor's in-place re-open
            stream, self._stream = self._stream, None
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception:
                    _log.warning("audio stop_stream() failed", exc_info=True)
                try:
                    stream.close()
                except Exception:
                    _log.warning("audio stream close() failed", exc_info=True)
            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:
                    _log.warning("PyAudio terminate() failed", exc_info=True)
                self._pa = None
        # stop() may run on a worker thread; the flush swaps the run out under
        # the continuity lock, which the PortAudio callback never takes.
        self._flush_continuity_run()
