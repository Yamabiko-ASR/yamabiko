"""NumPy-only compatibility boundary for faster-whisper."""
from __future__ import annotations

import importlib
import sys
from importlib import metadata
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any, Final, NoReturn

import numpy as np


SUPPORTED_FASTER_WHISPER_VERSION: Final[str] = "1.2.1"
_MARKER = "__yamabiko_numpy_audio_compat__"


class UnsupportedFasterWhisperVersion(RuntimeError):
    """Raised when the installed faster-whisper is outside this contract."""


def pad_or_trim(
    array: np.ndarray,
    length: int = 3000,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Match faster-whisper's NumPy padding/trimming helper."""
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)
    if array.shape[axis] < length:
        widths = [(0, 0)] * array.ndim
        widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, widths)
    return array


def decode_audio(
    input_file: object,
    sampling_rate: int = 16000,
    split_stereo: bool = False,
) -> NoReturn:
    """Reject file decoding; Yamabiko supplies captured NumPy arrays."""
    del input_file, sampling_rate, split_stereo
    raise RuntimeError(
        "Yamabiko supports captured NumPy audio arrays only; "
        "file decoding through faster-whisper/PyAV is intentionally disabled"
    )


def install_audio_compat() -> None:
    """Install a package-correct ``faster_whisper.audio`` replacement."""
    try:
        actual = metadata.version("faster-whisper")
    except metadata.PackageNotFoundError as exc:
        raise UnsupportedFasterWhisperVersion(
            "faster-whisper is not installed; install Yamabiko's pinned "
            "runtime dependencies"
        ) from exc
    if actual != SUPPORTED_FASTER_WHISPER_VERSION:
        raise UnsupportedFasterWhisperVersion(
            f"expected faster-whisper {SUPPORTED_FASTER_WHISPER_VERSION}, "
            f"got {actual}; update the compatibility module and its contract "
            "tests before changing the pin"
        )

    existing = sys.modules.get("faster_whisper.audio")
    if existing is not None and getattr(existing, _MARKER, False):
        return
    loaded = sorted(
        name
        for name in sys.modules
        if name == "faster_whisper" or name.startswith("faster_whisper.")
    )
    if loaded:
        raise RuntimeError(
            "faster-whisper was already imported before Yamabiko installed "
            f"its NumPy-only audio compatibility module: {loaded}"
        )

    module = ModuleType("faster_whisper.audio")
    module.__package__ = "faster_whisper"
    module.__loader__ = None
    module.__spec__ = ModuleSpec("faster_whisper.audio", loader=None)
    module.decode_audio = decode_audio
    module.pad_or_trim = pad_or_trim
    setattr(module, _MARKER, True)
    sys.modules[module.__name__] = module


def import_whisper_model() -> type[Any]:
    """Import ``WhisperModel`` only after installing the audio shim."""
    install_audio_compat()
    package = importlib.import_module("faster_whisper")
    # The child module is pre-seeded before the parent package import so
    # faster-whisper's module-level ``from faster_whisper.audio import ...``
    # resolves to the NumPy shim. Because the child was already present in
    # sys.modules, importlib does not necessarily perform its usual
    # parent-package attribute binding. Restore that package invariant
    # explicitly: callers must see the same module through both access paths.
    package.audio = sys.modules["faster_whisper.audio"]
    return package.WhisperModel
