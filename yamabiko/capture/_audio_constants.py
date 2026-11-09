"""Dependency-free audio constants shared across modules.

Deliberately imports NOTHING heavy (no pyaudiowpatch, no numpy): the first-run
wizard's audio page reads the silence floor from here, and the wizard must be
importable without dragging PortAudio into its import graph -- it guards
PortAudio init at call time and must not crash at module import on a partial /
broken bundle. capture/audio.py and wizard_pages/audio.py both import these so
there is one source of truth for the floor.
"""
from __future__ import annotations

# Amplitude-RMS floor (float32 [-1, 1]) below which a frame counts as silence.
# Real audio is ~0.05-0.5 RMS; digital silence is exactly 0.0; 1e-3 rides above
# denormal/dither noise without masking genuinely quiet content.
SILENCE_RMS_FLOOR: float = 1e-3

# Default seconds of sustained sub-floor audio (after audio was flowing) before
# the capture re-opens its stream in place. Operator-tunable via
# audio.idle_death_s. 90 s rides out a normal quiet stretch while recovering a
# genuinely stalled endpoint promptly. The action is a cheap in-place stream
# re-open, not a process restart, so an over-eager fire is harmless.
DEFAULT_IDLE_DEATH_S: int = 90


class LoopbackDeviceNotFound(RuntimeError):
    """The device pinned by ``audio.device`` is not among the loopback
    endpoints (unplugged, powered off, driver renamed it).

    Carries ``device_name`` as a FIELD. The startup-error dialog needs the
    name to tell the user which device vanished, and recovering it by
    re-parsing ``str(exc)`` is a trap: the message is built with ``!r``, and
    ``repr`` silently switches to double quotes for a name containing an
    apostrophe ("Dave's AirPods" -- a common Bluetooth default), so a
    single-quote parser misses exactly the class of device this error is
    most often raised for.

    Lives here, not in capture/audio.py, so ui/error_dialog.py can identify
    it without dragging PortAudio into the dialog's import graph (same
    reason the constants above are here).

    Subclasses RuntimeError: the previous raise site raised a bare
    RuntimeError and callers (and the classifier's keyword fallback) still
    match on that. The message text is unchanged for the same reason.
    """

    def __init__(self, device_name: str) -> None:
        super().__init__(f"Loopback device not found: {device_name!r}")
        self.device_name = device_name
