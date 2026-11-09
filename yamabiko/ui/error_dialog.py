"""The startup error window's data: the startup-failure cases and their copy.

``StartupError`` carries what the window shows (title, message, the "What to
try" sentences, the details) and, for a missing audio device, the Settings
page that fixes it. ``classify_startup_exception`` maps a pipeline startup
failure onto one of the cases (copy: the design record's Surface 6 table).

``show_startup_error`` logs the dialog line, then shows the "Report card"
window (``yamabiko.ui.error_window``, imported lazily: this module stays
Qt-free because the watchdog imports it at module level). When PySide6
cannot be imported, or the window fails, it falls back to a native Win32
message box with the same texts, the user name taken out. The window is
tested offscreen in ``tests/overlay_app/test_error_window.py`` and on the
native platform in ``tests/overlay_app/test_real_error_window.py``.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import PureWindowsPath

from yamabiko import paths
from yamabiko.capture._audio_constants import LoopbackDeviceNotFound
from yamabiko.lifecycle import startup_dialog
from yamabiko.models import BundledModelInvalid, ModelArtifactCorrupt
from yamabiko.ui import error_report

_log = logging.getLogger(__name__)

# Fallback parser for an untyped "Loopback device not found: <repr(name)>".
# The backreference makes the closing delimiter match the opening one, so both
# repr styles ('X' and "X") parse. See classify_startup_exception.
_LOOPBACK_DEVICE_NAME = re.compile(
    r"""loopback device not found:?\s*(['"])(.+)\1\s*\Z""",
    re.IGNORECASE | re.DOTALL,
)

# "Something went wrong while <words>." for the pipeline's stage labels
# (main.py); any other stage reads "starting Yamabiko". The stage itself
# stays in the details ("Stage: ...").
_STAGE_WORDS = {
    "config load": "loading your settings",
    "VAD init": "starting speech detection",
    "ASR init": "starting speech recognition",
    "translator sanity check": "starting the translator",
    "audio capture": "starting audio capture",
}
# One "What to try" item per sentence: split after . ! or ? and whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_DRIVE_LETTER = re.compile(r"[A-Za-z]:")


@dataclass(frozen=True)
class StartupError:
    title: str
    message: str
    suggested_fix: str
    details: str
    # The Settings page ("audio") the window's "Open Settings" opens; None
    # shows no such button. Only audio_device_gone sets it.
    settings_page: str | None = None

    def what_to_try(self) -> list[str]:
        """``suggested_fix`` as the window's "What to try" items, one per
        sentence."""
        return [part for part in _SENTENCE_END.split(self.suggested_fix.strip()) if part]

    # --- factories for each canonical case ---

    @classmethod
    def gpu_unavailable(cls, *, details: str) -> "StartupError":
        return cls(
            title="No NVIDIA graphics card found",
            message=(
                "Yamabiko needs an NVIDIA graphics card and couldn't find one "
                "it can use."
            ),
            suggested_fix=(
                "Install the latest NVIDIA driver, then restart your PC. "
                "Yamabiko works with RTX 20 series cards and newer that have "
                "8 GB of video memory or more."
            ),
            details=details,
        )

    @classmethod
    def no_audio_device(cls, *, details: str) -> "StartupError":
        return cls(
            title="No speakers or headphones found",
            message=(
                "Yamabiko listens to what your PC plays, and it couldn't find "
                "anything to listen to."
            ),
            suggested_fix="Connect speakers or headphones, then start Yamabiko again.",
            details=details,
        )

    @classmethod
    def audio_device_gone(cls, *, device_name: str, details: str) -> "StartupError":
        # Distinct from no_audio_device: loopback capture works in general,
        # but the SPECIFIC device the user pinned (audio.device) is gone -- a
        # Bluetooth headset powered off, a monitor unplugged. The title names
        # it as the Settings list shows it, without the " [Loopback]" suffix
        # the saved name carries, and "Open Settings" opens its picker.
        name = device_name.removesuffix(" [Loopback]")
        return cls(
            title=f"{name} isn't connected",
            message="Yamabiko is set to listen to it.",
            suggested_fix=(
                "Reconnect it, then start Yamabiko again. Or pick another "
                "device in Settings."
            ),
            details=details,
            settings_page="audio",
        )

    @classmethod
    def llama_failed(cls, *, details: str) -> "StartupError":
        return cls(
            title="The translator didn't start",
            message="It took too long to load.",
            suggested_fix=(
                "Close apps that use a lot of video memory, such as games, "
                "then start Yamabiko again."
            ),
            details=details,
        )

    @classmethod
    def models_missing(cls, *, details: str) -> "StartupError":
        # No tray-menu setup action exists: deleting state.json makes the
        # next launch re-enter the first-run setup, which downloads only the
        # missing files. A reinstall keeps state.json, so it is not offered.
        return cls(
            title="Some model files are missing",
            message="Yamabiko can't start without them.",
            suggested_fix=(
                "To download them again, delete state.json in "
                "%LOCALAPPDATA%\\Yamabiko, then start Yamabiko. Setup runs "
                "again and gets only the missing files."
            ),
            details=details,
        )

    @classmethod
    def model_file_corrupt(cls, *, path: str, details: str) -> "StartupError":
        # Distinct from models_missing: re-running the first-run setup
        # cannot repair a damaged cached file, because hub returns the
        # existing file for a pinned commit instead of downloading it
        # again. Deleting the named file is what makes the next start
        # fetch a fresh copy. The message names the file; its full path
        # stays in the details (redacted in the report).
        return cls(
            title="A model file is damaged",
            message=f"{PureWindowsPath(path).name} didn't pass its check.",
            suggested_fix=(
                "Delete it, then start Yamabiko again to download a fresh "
                "copy. This needs an internet connection."
            ),
            details=details,
        )

    @classmethod
    def installed_file_damaged(cls, *, path: str, details: str) -> "StartupError":
        # A model shipped inside the install (the sound tagger), not a
        # download: deleting it or re-running setup cannot bring it back,
        # only a reinstall restores it.
        return cls(
            title="A Yamabiko file is damaged",
            message=f"{PureWindowsPath(path).name} is missing or damaged.",
            suggested_fix="Reinstall Yamabiko. Your settings and models stay.",
            details=details,
        )

    @classmethod
    def disk_space(cls, *, details: str, drive: str | None = None) -> "StartupError":
        # ``drive`` ("C:") names where the download ran out of room; without
        # one the copy names no drive.
        if drive:
            message = f"The model download ran out of space on drive {drive}."
            fix = f"Free up about 7 GB on drive {drive}, then start Yamabiko again."
        else:
            message = "The model download ran out of disk space."
            fix = "Free up about 7 GB, then start Yamabiko again."
        return cls(
            title="Not enough disk space",
            message=message,
            suggested_fix=fix,
            details=details,
        )


def _download_drive(exc: BaseException) -> str | None:
    """The drive letter ("C:") the failed download wrote to: the OSError's
    file name, else HF_HOME, else Yamabiko's data folder; the first of them
    that names a drive decides. None for a UNC share or no drive at all."""
    for candidate in (getattr(exc, "filename", None), os.environ.get("HF_HOME"),
                      str(paths.app_data_dir())):
        if not candidate:
            continue
        drive = PureWindowsPath(os.fsdecode(candidate)).drive
        if drive:
            return drive.upper() if _DRIVE_LETTER.fullmatch(drive) else None
    return None


def _dll_load_failure(exc: BaseException) -> bool:
    """True for a DLL that failed to load: ctypes raises FileNotFoundError
    ("Could not find module '...\\ctranslate2.dll' (or one of its
    dependencies)") for a missing or damaged DLL or dependency, and Windows
    reports one as error 126. That is a damaged install, not a missing
    model: deleting state.json can't fix it."""
    if getattr(exc, "winerror", None) == 126:
        return True
    return ".dll" in f"{exc} {getattr(exc, 'filename', None) or ''}".lower()


def _file_not_found_in_chain(exc: BaseException) -> bool:
    """True when ``exc`` or an exception it was raised from (``__cause__``,
    or ``__context__`` unless suppressed) is a FileNotFoundError, such as
    huggingface_hub's LocalEntryNotFoundError for a model that is not in the
    cache. A DLL that failed to load does not count (it stays generic)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, FileNotFoundError) and not _dll_load_failure(current):
            return True
        seen.add(id(current))
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return False


def classify_startup_exception(stage: str, exc: BaseException) -> StartupError:
    """Pattern-match an exception into one of the tailored StartupError
    cases, or fall back to a generic one if nothing matches.

    The typed integrity failures come first, then the GPU keywords, a
    missing file anywhere in the exception chain and the pinned loopback
    device, then keyword lookups on the message. When nothing matches, the
    generic case names the stage in plain words and keeps the full
    traceback in the details.

    ``stage`` is a short label for the failing pipeline step ("ASR init",
    "translator sanity check", "audio capture", ...). It is prefixed to the
    details blob so the report says what failed.
    """
    import traceback
    if isinstance(exc, BaseException):
        tb = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    else:
        tb = str(exc)
    details = f"Stage: {stage}\n\n{tb}"
    msg = (str(exc) or "").lower()
    # Order matters: more specific patterns first. The typed integrity
    # failure leads because its message embeds a user path, which the
    # keyword arms below could match by accident.
    if isinstance(exc, ModelArtifactCorrupt):
        return StartupError.model_file_corrupt(path=exc.path, details=details)
    if isinstance(exc, BundledModelInvalid):
        return StartupError.installed_file_damaged(path=exc.path, details=details)
    if any(k in msg for k in (
        "cublas", "cudnn", "nvrtc", "cuda",
        "no compatible gpu", "cuda_not_available",
        "torch.cuda", "device 'cuda'",
    )):
        return StartupError.gpu_unavailable(details=details)
    # A model missing from the cache: huggingface_hub raises
    # LocalEntryNotFoundError (a FileNotFoundError) with no keyword the arms
    # below know (frozen probe, plan phase 6), sometimes wrapped by the
    # loader. After the GPU arm: Windows reports a CUDA library that fails to
    # load as a FileNotFoundError naming it, without its extension when it was
    # loaded by bare name ("Could not find module 'cublas64_12' (or one of
    # its dependencies)"), which the DLL rule below does not skip. Any other
    # DLL that fails to load is skipped by the walk and stays generic (a
    # damaged install, not a missing model).
    if _file_not_found_in_chain(exc):
        return StartupError.models_missing(details=details)
    # Pinned-device-gone before the generic audio arm: the CONFIGURED device
    # vanished, while other loopback endpoints may be fine. The name comes off
    # the typed exception as a FIELD -- never re-parsed out of the message.
    if isinstance(exc, LoopbackDeviceNotFound):
        return StartupError.audio_device_gone(
            device_name=exc.device_name, details=details,
        )
    # Fallback for the same failure arriving as a plain RuntimeError (an older
    # raise site, or a message crossing a boundary that dropped the type).
    # Accepts EITHER quote style: the message is built with !r, and repr uses
    # double quotes when the name contains an apostrophe. A name containing
    # both quote styles still parses imperfectly here -- which is precisely why
    # the typed path above exists and is the one the product actually takes.
    name_match = _LOOPBACK_DEVICE_NAME.search(str(exc))
    if name_match:
        return StartupError.audio_device_gone(
            device_name=name_match.group(2), details=details,
        )
    # Deliberate fallback, NOT dead code: a loopback failure with no parsable
    # device name (the "default" device path raises "No WASAPI loopback device
    # found for default output", which names no device) still deserves the
    # audio-specific dialog rather than the generic "couldn't start" one.
    if any(k in msg for k in (
        "no wasapi", "no audio device", "no loopback",
        "wasapi loopback device not found",
        "loopback device not found",
    )):
        return StartupError.no_audio_device(details=details)
    if any(k in msg for k in (
        "connection refused", "connecterror",
        "localhost:8090", "127.0.0.1:8090",
        "is llama-server running", "llama-server",
    )):
        return StartupError.llama_failed(details=details)
    if any(k in msg for k in (
        "no space left", "disk full", "errno 28",
        "insufficient disk",
    )):
        return StartupError.disk_space(details=details, drive=_download_drive(exc))
    if any(k in msg for k in (
        "file not found", "no such file",
        "filenotfounderror", "model.bin", ".gguf",
        ".onnx", "cached_no_exist",
    )):
        return StartupError.models_missing(details=details)
    # Default: generic fallback naming the stage in plain words.
    words = _STAGE_WORDS.get(stage, "starting Yamabiko")
    return StartupError(
        title="Yamabiko couldn't start",
        message=f"Something went wrong while {words}.",
        suggested_fix=(
            "Start Yamabiko again. If it keeps happening, copy the report "
            "and send it to the developer."
        ),
        details=details,
    )


def _show_native_message_box(err: StartupError) -> None:
    """Render ``err`` through Win32 MessageBoxW (MB_ICONERROR): the window's
    texts, the log folder and the details, the user name taken out of the
    message, what to try and the details (a hand-built message can carry
    exception text with a path). The caption keeps the title as it is."""
    import ctypes

    redact = error_report.redact_user
    message = (
        f"{redact(err.message)}\n\n{redact(err.suggested_fix)}\n\n"
        f"Logs are in %LOCALAPPDATA%\\Yamabiko\\logs.\n\n"
        f"{redact(err.details)}"
    )
    title = f"Yamabiko: {err.title}"
    with startup_dialog.held():
        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)


def show_startup_error(err: StartupError) -> None:
    """Show ``err`` in the startup error window and return when it is
    closed, together with a Settings window it opened. Qt is imported
    lazily, so headless callers need no Qt; a QApplication is created when
    none exists.
    """
    # Verbatim copy of the dialog text into the log BEFORE any Qt work:
    # users paraphrase dialogs in bug reports, the log line is ground
    # truth -- and it must exist even if Qt fails to import or hangs
    # mid-render. run_error_window reads the window's log tail before
    # anything else logs, so the tail ends here.
    _log.error("startup error dialog: %s | %s", err.title, err.message)
    try:
        from yamabiko.ui.error_window import run_error_window
    except ImportError:
        # A broken or missing Qt install (damaged bundle, antivirus
        # quarantine of a Qt DLL) must still leave the user a dialog,
        # not a silent exit. The native message box needs only user32.
        if not sys.platform.startswith("win"):
            raise
        _log.warning("PySide6 unavailable; using a native message box", exc_info=True)
        _show_native_message_box(err)
        return
    # Hold the startup-dialog marker for the window's whole life, a
    # Settings window it opens included: the watchdog's liveness probe
    # would otherwise read this blocked process as hung and kill it under
    # the user.
    with startup_dialog.held():
        try:
            run_error_window(err)
        except Exception:
            # The window must never cost the user the dialog or the caller
            # its exit code: every caller exits 43 right after this returns,
            # and an exception here would crash the pipeline into a restart
            # of the same failure. The native box needs only user32 (its own
            # held() just reopens the marker this one holds).
            if not sys.platform.startswith("win"):
                raise
            _log.exception("startup error window failed; using a native message box")
            _show_native_message_box(err)
