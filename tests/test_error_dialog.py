"""Error dialog tests: StartupError's copy, its what-to-try items, the
classifier and the native fallback (Qt-free).

The window itself (show_startup_error's Qt branch) is tested offscreen in
tests/overlay_app/test_error_window.py and on the native platform in
tests/overlay_app/test_real_error_window.py; the report data in
tests/test_error_report.py.
"""
from __future__ import annotations

import pytest

from yamabiko.ui import error_dialog as ed


_COPY = [
    ("gpu_unavailable", {}, "No NVIDIA graphics card found",
     "Yamabiko needs an NVIDIA graphics card and couldn't find one it can use.",
     "Install the latest NVIDIA driver, then restart your PC. Yamabiko works with RTX 20 "
     "series cards and newer that have 8 GB of video memory or more.", None),
    ("no_audio_device", {}, "No speakers or headphones found",
     "Yamabiko listens to what your PC plays, and it couldn't find anything to listen to.",
     "Connect speakers or headphones, then start Yamabiko again.", None),
    ("audio_device_gone", {"device_name": "Headphones (USB Audio) [Loopback]"},
     "Headphones (USB Audio) isn't connected", "Yamabiko is set to listen to it.",
     "Reconnect it, then start Yamabiko again. Or pick another device in Settings.", "audio"),
    ("llama_failed", {}, "The translator didn't start", "It took too long to load.",
     "Close apps that use a lot of video memory, such as games, then start Yamabiko again.", None),
    ("models_missing", {}, "Some model files are missing", "Yamabiko can't start without them.",
     "To download them again, delete state.json in %LOCALAPPDATA%\\Yamabiko, then start "
     "Yamabiko. Setup runs again and gets only the missing files.", None),
    ("model_file_corrupt",
     {"path": "C:\\Users\\Ann\\AppData\\Local\\Yamabiko\\hf-cache\\silero_vad.onnx"},
     "A model file is damaged", "silero_vad.onnx didn't pass its check.",
     "Delete it, then start Yamabiko again to download a fresh copy. This needs an internet "
     "connection.", None),
    ("installed_file_damaged", {"path": "C:\\Program Files\\Yamabiko\\_internal\\tagger.onnx"},
     "A Yamabiko file is damaged", "tagger.onnx is missing or damaged.",
     "Reinstall Yamabiko. Your settings and models stay.", None),
    ("disk_space", {"drive": "D:"}, "Not enough disk space",
     "The model download ran out of space on drive D:.",
     "Free up about 7 GB on drive D:, then start Yamabiko again.", None),
    ("disk_space", {}, "Not enough disk space", "The model download ran out of disk space.",
     "Free up about 7 GB, then start Yamabiko again.", None),
]


@pytest.mark.parametrize("factory, kwargs, title, message, fix, page", _COPY, ids=[
    "no-gpu", "no-audio", "device-gone", "translator", "models-missing", "model-damaged",
    "installed-damaged", "disk-full-drive", "disk-full-no-drive",
])
def test_each_case_carries_the_spec_copy(factory, kwargs, title, message, fix, page):
    """The design record's Surface 6 copy table, verbatim: title, message,
    What to try; only the missing device opens Settings (Audio & model).
    The model and installed file cases name the file, not its folder.

    Seeded defects: any copy edit, the " [Loopback]" suffix kept in the
    device title, the full path in a message, or settings_page on another
    case -> red."""
    err = getattr(ed.StartupError, factory)(details="the details", **kwargs)
    assert (err.title, err.message, err.suggested_fix, err.settings_page) == (title, message, fix, page)
    assert err.details == "the details"


def test_models_missing_fix_points_at_state_json_not_a_tray_menu():
    """The recovery instruction must point at the supported re-run path --
    delete state.json so the next launch re-enters the first-run wizard --
    and must NOT send the user hunting for a tray-menu setup action that
    does not exist (an earlier version claimed one; users dead-ended at
    the dialog).

    Seeded-defect checks (each one line):
      - reword suggested_fix to drop the state.json path -> first
        assert red.
      - reintroduce 'use the tray menu's Setup action' phrasing ->
        second assert red.
    """
    err = ed.StartupError.models_missing(details="FileNotFoundError: foo.ct2")
    assert "state.json" in err.suggested_fix, (
        f"models_missing.suggested_fix no longer names the state.json "
        f"re-run path; got: {err.suggested_fix!r}"
    )
    assert "tray" not in err.suggested_fix.lower(), (
        f"models_missing.suggested_fix claims a tray action again -- no "
        f"tray-menu setup action exists; got: "
        f"{err.suggested_fix!r}"
    )


def test_what_to_try_is_one_item_per_sentence():
    """L2: the window shows suggested_fix as one dash item per sentence,
    split after ".", "!" or "?" and whitespace.

    Seeded defects: split on every "." (breaks "state.json") or only on
    ". " (misses "!" and "?") -> red."""
    assert ed.StartupError.models_missing(details="d").what_to_try() == [
        "To download them again, delete state.json in %LOCALAPPDATA%\\Yamabiko, then start Yamabiko.",
        "Setup runs again and gets only the missing files.",
    ]
    err = ed.StartupError(title="T", message="M", details="",
                          suggested_fix="Restart Yamabiko! Still broken? Send the report.")
    assert err.what_to_try() == ["Restart Yamabiko!", "Still broken?", "Send the report."]


@pytest.mark.skipif(
    not __import__("sys").platform.startswith("win"),
    reason="the native fallback is Win32 MessageBoxW",
)
def test_show_startup_error_without_pyside6_uses_native_message_box(monkeypatch):
    """A bundle whose Qt cannot import (damaged install, quarantined DLL)
    must still show the user the error through MessageBoxW, not raise
    ImportError out of the error path. Same texts as the window, with the
    user name taken out of the message, what to try and the details (a
    hand-built message can carry exception text with a path, ruling I2);
    the caption keeps the title as it is.

    Seeded defects: drop the ImportError fallback -> ImportError escapes
    and MessageBoxW is never called; show the raw message, what to try or
    details -> the user's folder is in the box -> red.
    """
    import ctypes
    import dataclasses
    import sys as _sys

    # A None entry makes the import raise ImportError ("halted; None in
    # sys.modules"), the same exception a missing or broken Qt raises. The
    # window module is dropped too, so its import runs (and fails) again,
    # as on a first error in a fresh process.
    monkeypatch.setitem(_sys.modules, "PySide6", None)
    monkeypatch.setitem(_sys.modules, "PySide6.QtWidgets", None)
    monkeypatch.delitem(_sys.modules, "yamabiko.ui.error_window", raising=False)
    monkeypatch.setenv("USERPROFILE", r"C:\Users\Tester Person")
    monkeypatch.setenv("USERNAME", "Tester Person")
    calls: list = []
    monkeypatch.setattr(
        ctypes.windll.user32, "MessageBoxW",
        lambda hwnd, text, caption, flags: calls.append((hwnd, text, caption, flags)) or 1,
    )

    err = dataclasses.replace(
        ed.StartupError.gpu_unavailable(
            details=r"CUDA init failed: C:\Users\Tester Person\cuda\cublas64_12.dll"),
        message=r"The pipeline process could not be spawned: C:\Users\Tester Person\yb\yamabiko.exe",
        suggested_fix=r"Reinstall Yamabiko to C:\Users\Tester Person\yb, then restart.",
    )
    ed.show_startup_error(err)

    assert len(calls) == 1, f"MessageBoxW calls: {calls}"
    hwnd, text, caption, flags = calls[0]
    assert hwnd is None and flags == 0x10
    assert caption == "Yamabiko: No NVIDIA graphics card found"
    assert r"The pipeline process could not be spawned: %USERPROFILE%\yb\yamabiko.exe" in text
    assert r"Reinstall Yamabiko to %USERPROFILE%\yb, then restart." in text
    assert r"CUDA init failed: %USERPROFILE%\cuda\cublas64_12.dll" in text
    assert "tester person" not in text.lower()


@pytest.mark.skipif(
    not __import__("sys").platform.startswith("win"),
    reason="the native fallback is Win32 MessageBoxW",
)
def test_show_startup_error_falls_back_to_the_native_box_when_the_window_fails(monkeypatch, caplog):
    """Ruling I3: every caller exits 43 right after show_startup_error
    returns. An exception out of the window (a fact read, a font, the
    build) would instead crash the pipeline into a restart of the same
    failure, and the user would see nothing. The failure is logged with its
    traceback and the native box shows the error.

    Seeded defect: no except around run_error_window -> RuntimeError
    escapes and MessageBoxW is never called -> red."""
    import ctypes
    import logging
    import sys as _sys
    import types

    def broken(err):
        raise RuntimeError("the window broke")

    # A stand-in window module whose run_error_window fails (this file
    # never builds a Qt window).
    window_module = types.ModuleType("yamabiko.ui.error_window")
    window_module.run_error_window = broken
    monkeypatch.setitem(_sys.modules, "yamabiko.ui.error_window", window_module)
    captions: list[str] = []
    monkeypatch.setattr(
        ctypes.windll.user32, "MessageBoxW",
        lambda hwnd, text, caption, flags: captions.append(caption) or 1,
    )
    with caplog.at_level(logging.ERROR, logger="yamabiko.ui.error_dialog"):
        ed.show_startup_error(ed.StartupError.gpu_unavailable(details="d"))
    assert captions == ["Yamabiko: No NVIDIA graphics card found"]
    assert any(record.exc_info and "the window broke" in str(record.exc_info[1])
               for record in caplog.records), caplog.text


# ---------------------------------------------------------------------------
# classify_startup_exception
#
# Before this function existed, the 5 factory classmethods above were
# dead code -- nothing pattern-matched an exception into them. These
# tests lock in the pattern-match so each classifier-path keyword
# remains wired. If someone renames a StartupError factory or removes
# a keyword, these tests catch it.
# ---------------------------------------------------------------------------


def _classify(stage: str, exc: BaseException) -> ed.StartupError:
    return ed.classify_startup_exception(stage, exc)


def test_classify_cuda_error_to_gpu_unavailable():
    exc = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
    err = _classify("ASR init", exc)
    assert err.title == "No NVIDIA graphics card found"
    assert "cublas" in err.details.lower()


def test_classify_cudnn_error_to_gpu_unavailable():
    exc = RuntimeError("cuDNN error: CUDNN_STATUS_NOT_INITIALIZED")
    err = _classify("ASR warmup", exc)
    assert err.title == "No NVIDIA graphics card found"


def test_classify_no_wasapi_to_audio():
    exc = RuntimeError("No WASAPI loopback device found for default output")
    err = _classify("capture start", exc)
    assert err.title == "No speakers or headphones found"


def test_classify_connection_refused_to_llama_failed():
    exc = ConnectionError("[WinError 10061] No connection could be made because the target machine actively refused it (127.0.0.1:8090)")
    err = _classify("translator sanity check", exc)
    assert err.title == "The translator didn't start"


def test_classify_llama_server_message_to_llama_failed():
    exc = RuntimeError("translator sanity check failed (is llama-server running?): HTTPConnectionPool(host='localhost', port=8090)")
    err = _classify("translator sanity check", exc)
    assert err.title == "The translator didn't start"


def test_classify_missing_model_file_to_models_missing():
    exc = FileNotFoundError(2, "No such file or directory", "model.bin")
    err = _classify("ASR init", exc)
    assert err.title == "Some model files are missing"


def test_classify_gguf_missing_to_models_missing():
    exc = FileNotFoundError("shisa-v2.1-qwen3-8B-UD-Q4_K_XL.gguf not found in cache")
    err = _classify("llama spawn", exc)
    assert err.title == "Some model files are missing"


def test_classify_disk_full_to_disk_space():
    exc = OSError(28, "No space left on device", "/tmp/download")
    err = _classify("HF download", exc)
    assert err.title == "Not enough disk space"


def test_disk_full_names_the_drive_the_download_wrote_to(monkeypatch):
    """L3: the drive comes from the failed write's file name, else HF_HOME
    (where downloads land), else Yamabiko's data folder; a UNC share names
    no drive.

    Seeded defects: always HF_HOME's drive, always "C:", a UNC share's
    "\\\\server\\share" shown as a drive, or no data-folder fallback -> red."""
    monkeypatch.setenv("HF_HOME", "E:\\hf-cache")
    full = OSError(28, "No space left on device", "D:\\hf-cache\\blobs\\x.incomplete")
    assert _classify("ASR init", full).message == "The model download ran out of space on drive D:."
    no_name = OSError(28, "No space left on device")
    assert _classify("ASR init", no_name).message == "The model download ran out of space on drive E:."
    monkeypatch.setenv("HF_HOME", "\\\\server\\share\\hf-cache")
    assert _classify("ASR init", no_name).message == "The model download ran out of disk space."
    monkeypatch.delenv("HF_HOME")
    monkeypatch.setenv("YAMABIKO_APPDATA", "F:\\yb")
    assert _classify("ASR init", no_name).message == "The model download ran out of space on drive F:."


def test_model_missing_from_the_cache_maps_to_models_missing_even_wrapped(tmp_path):
    """Lead ruling (frozen probe): a model missing from the cache raises
    huggingface_hub's LocalEntryNotFoundError, a FileNotFoundError whose
    message holds no keyword the arms know, so it read "Yamabiko couldn't
    start". Any FileNotFoundError in the exception chain is the models case
    (here the real hub error, wrapped by a loader). A Windows "Could not find
    module" for a CUDA library loaded without its extension is also a
    FileNotFoundError (one the DLL rule does not skip) and stays the GPU case.

    Seeded defects: no chain arm (the generic case) or one that checks only
    the outer exception -> first assert red; the chain arm before the GPU
    keywords -> second assert red."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    with pytest.raises(LocalEntryNotFoundError) as missing:
        snapshot_download("Systran/faster-whisper-large-v3", cache_dir=tmp_path, local_files_only=True)
    try:
        raise RuntimeError("ASR model load failed") from missing.value
    except RuntimeError as wrapped:
        err = _classify("ASR init", wrapped)
    assert err.title == "Some model files are missing"
    dll = FileNotFoundError("Could not find module 'cublas64_12' (or one of its dependencies). "
                            "Try using the full path with constructor syntax.")
    assert _classify("ASR init", dll).title == "No NVIDIA graphics card found"


def test_a_dll_that_fails_to_load_is_not_a_missing_model():
    """Ruling M1: ctypes reports a missing or damaged DLL (or one of its
    dependencies) as a FileNotFoundError naming the DLL. A damaged or
    quarantined ctranslate2.dll or onnxruntime.dll is no missing model:
    "delete state.json" cannot fix it, so it stays the generic case, named
    by the message (the ctypes form) or by the file name.

    Seeded defect: the chain arm without its DLL check -> "Some model files
    are missing" -> red."""
    by_message = FileNotFoundError(
        "Could not find module 'C:\\Program Files\\Yamabiko\\_internal\\ctranslate2\\ctranslate2.dll'"
        " (or one of its dependencies). Try using the full path with constructor syntax.")
    by_name = FileNotFoundError(
        2, "The specified module could not be found",
        "C:\\Program Files\\Yamabiko\\_internal\\onnxruntime\\capi\\onnxruntime.dll")
    for failure in (by_message, by_name):
        try:
            raise RuntimeError("ASR model load failed") from failure
        except RuntimeError as wrapped:
            err = _classify("ASR init", wrapped)
        assert err.title == "Yamabiko couldn't start", failure


def test_damaged_bundled_sound_tagger_maps_to_reinstall_dialog(tmp_path):
    """A damaged bundled tagger stops startup with the reinstall dialog.

    The file ships with the installer, so the download-oriented dialogs
    ("delete it and start again", "re-run setup") cannot repair it. Without
    the digest check ONNX Runtime fails on the bytes with a parse error that
    lands in another dialog; without the classify branch the typed error
    falls through to the generic one.
    """
    from yamabiko.models import BundledModelInvalid
    from yamabiko.vad.sound_tagger import SoundTagger

    damaged = tmp_path / "efficientsed-fmn10-tf256-kd.onnx"
    damaged.write_bytes(b"not the pinned graph")

    with pytest.raises(BundledModelInvalid) as raised:
        SoundTagger(damaged)
    assert raised.value.path == str(damaged.resolve())

    err = _classify("VAD init", raised.value)
    assert err.title == "A Yamabiko file is damaged"
    assert err.message == "efficientsed-fmn10-tf256-kd.onnx is missing or damaged."
    assert "Reinstall Yamabiko" in err.suggested_fix



@pytest.mark.parametrize("stage, words", [
    ("config load", "loading your settings"),
    ("VAD init", "starting speech detection"),
    ("ASR init", "starting speech recognition"),
    ("translator sanity check", "starting the translator"),
    ("audio capture", "starting audio capture"),
    ("pipeline init", "starting Yamabiko"),
])
def test_generic_failure_names_the_stage_in_plain_words(stage, words):
    """L3: "Something went wrong while <the part in words>." for each
    pipeline stage, "starting Yamabiko" for any other; the stage label and
    the traceback stay in the details.

    Seeded defects: the raw stage label in the message, a missing map entry,
    or no fallback -> red."""
    err = _classify(stage, ValueError("something really weird happened"))
    assert err.title == "Yamabiko couldn't start"
    assert err.message == f"Something went wrong while {words}."
    assert err.suggested_fix == (
        "Start Yamabiko again. If it keeps happening, copy the report and send it to the developer.")
    assert err.details.startswith(f"Stage: {stage}\n")
    assert "ValueError" in err.details


@pytest.mark.parametrize("device_name", [
    "Headphones (yamabiko) [Loopback]",   # no quote characters
    "Dave's AirPods [Loopback]",          # apostrophe -> repr() switches to "..."
    'Bob\'s "Good" Buds [Loopback]',      # both quote styles -> repr escapes
])
def test_classify_typed_loopback_gone_names_device_for_any_quote_style(device_name):
    """The pinned-device-gone dialog must name the missing device and point at
    the Settings audio picker, for EVERY name a real endpoint can have. The
    generic "No loopback devices were found" copy is factually wrong here:
    other loopback devices usually still exist; only the pinned one vanished.

    Field regression: a Bluetooth headset powered off mid-session and the
    dialog claimed no loopback devices were found and suggested enabling
    Stereo Mix, neither of which named the real problem or the real fix.

    This is the CONSUMER half of the round trip whose PRODUCER half is
    tests/capture/test_audio.py::test_missing_pinned_device_raises_typed_error_carrying_the_exact_name.
    It is driven by the REAL exception type the real raise site constructs, not
    a hand-written message string: an earlier revision of this fix recovered the
    name by regex-parsing str(exc) with a single-quote pattern, which silently
    failed for the apostrophe names above (repr() switches to double quotes) and
    so fell back to the exact wrong dialog it was written to replace. A test that
    hand-writes the message can never see that.

    Seeded-defect drill (verified red 2026-07-12): comment out the
    `isinstance(exc, LoopbackDeviceNotFound)` arm in classify_startup_exception
    -> the both-quote-styles case goes red, because the message-parsing fallback
    cannot recover a name whose repr() backslash-escapes its apostrophe. The
    apostrophe-only case survives that single revert (the fallback's
    either-quote backreference still recovers it), which is the point of keeping
    two layers: reverting BOTH -- the isinstance arm and the backreference --
    reproduces the original defect and takes the apostrophe case red too.
    """
    from yamabiko.capture._audio_constants import LoopbackDeviceNotFound

    err = _classify("audio capture", LoopbackDeviceNotFound(device_name))

    assert err.title == f"{device_name.removesuffix(' [Loopback]')} isn't connected", (
        f"device-gone dialog must name the missing device verbatim; got "
        f"title={err.title!r}"
    )
    assert err.settings_page == "audio"
    assert "Settings" in err.suggested_fix, (
        f"fix must point at the Settings audio picker; got {err.suggested_fix!r}"
    )
    assert err.message == "Yamabiko is set to listen to it."


@pytest.mark.parametrize("raw, name", [
    ("Loopback device not found: 'Speakers (Realtek)'", "Speakers (Realtek)"),  # repr single-quoted
    ('Loopback device not found: "Dave\'s AirPods"', "Dave's AirPods"),         # repr double-quoted
])
def test_classify_untyped_loopback_gone_message_still_names_device(raw, name):
    """Fallback path: the same failure arriving as a plain RuntimeError (an
    older raise site, or a message that crossed a boundary and lost its type)
    still yields the device-gone dialog, for BOTH repr quote styles.

    Seeded-defect check: drop the backreference from _LOOPBACK_DEVICE_NAME
    (hardcode a single-quote delimiter) -> the double-quoted case falls through
    to the generic dialog -> red.
    """
    err = _classify("audio capture", RuntimeError(raw))
    assert err.title == f"{name} isn't connected", (
        f"untyped device-gone message must still be recognised; got {err.title!r}"
    )


def test_classify_loopback_gone_without_parsable_name_falls_back_to_no_audio():
    """A loopback failure that names no device (the "default" device path
    raises "No WASAPI loopback device found for default output") degrades to
    the generic no-audio dialog, never a device-gone dialog with an empty name.

    Seeded-defect check: loosen the device-name extraction to match an
    unquoted tail (r"not found:?\\s*(.*)") -> this message classifies as
    a device-gone dialog with an empty device name -> red.
    """
    exc = RuntimeError("Loopback device not found")
    err = _classify("audio capture", exc)
    assert err.title == "No speakers or headphones found", (
        f"unparsable-name form must use the generic audio dialog; got {err.title!r}"
    )
