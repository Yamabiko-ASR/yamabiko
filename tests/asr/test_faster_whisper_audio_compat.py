"""Contract tests for Yamabiko's NumPy-only faster-whisper import seam."""
from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from yamabiko.asr import faster_whisper_audio_compat as compat


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_fresh_python(source: str) -> subprocess.CompletedProcess[str]:
    """Run an import-order probe in an unpoisoned Python interpreter."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_child_succeeded(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"child exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def test_pad_or_trim_pads_requested_axis_and_preserves_dtype():
    source = np.arange(6, dtype=np.float32).reshape(2, 3)

    result = compat.pad_or_trim(source, 5, axis=1)

    assert result.dtype == source.dtype
    np.testing.assert_array_equal(result[:, :3], source)
    np.testing.assert_array_equal(
        result[:, 3:], np.zeros((2, 2), dtype=np.float32),
    )


def test_pad_or_trim_trims_requested_axis():
    source = np.arange(12).reshape(3, 4)

    result = compat.pad_or_trim(source, 2, axis=0)

    np.testing.assert_array_equal(result, source[:2])


def test_decode_audio_rejects_file_input_with_numpy_only_contract():
    with pytest.raises(RuntimeError, match="NumPy audio arrays"):
        compat.decode_audio("sample.wav")


def test_import_whisper_model_installs_audio_shim_without_importing_av():
    result = _run_fresh_python(
        """
        import importlib.abc
        import sys


        class RejectPyAv(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "av" or fullname.startswith("av."):
                    raise AssertionError(f"PyAV import attempted: {fullname}")
                return None


        sys.meta_path.insert(0, RejectPyAv())

        from yamabiko.asr import faster_whisper_audio_compat as compat

        whisper_model = compat.import_whisper_model()

        import faster_whisper
        import faster_whisper.transcribe as transcribe

        assert whisper_model is transcribe.WhisperModel
        assert faster_whisper.audio is sys.modules["faster_whisper.audio"]
        assert transcribe.decode_audio is compat.decode_audio
        assert transcribe.pad_or_trim is compat.pad_or_trim
        assert sys.modules["faster_whisper.audio"].__package__ == "faster_whisper"
        assert (
            sys.modules["faster_whisper.audio"].__spec__.name
            == "faster_whisper.audio"
        )
        assert not any(
            name == "av" or name.startswith("av.") for name in sys.modules
        )
        print("WHISPER_IMPORT_OK")
        """
    )

    _assert_child_succeeded(result)
    assert "WHISPER_IMPORT_OK" in result.stdout


def test_runtime_requirement_pin_matches_supported_compat_version():
    """The exact runtime pin and the compatibility contract cannot drift.

    Seeded defect: change the requirements entry to
    ``faster-whisper==1.2.0`` while leaving the compatibility constant at
    1.2.1; this assertion fails with both values visible.
    """
    entries = [
        line.strip()
        for line in (
            _REPO_ROOT / "yamabiko" / "requirements.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    faster_whisper_entries = [
        entry
        for entry in entries
        if entry.lower().startswith("faster-whisper")
    ]

    assert faster_whisper_entries == [
        f"faster-whisper=={compat.SUPPORTED_FASTER_WHISPER_VERSION}"
    ]


def test_install_audio_compat_rejects_unsupported_installed_version():
    result = _run_fresh_python(
        """
        from yamabiko.asr import faster_whisper_audio_compat as compat

        compat.metadata.version = lambda distribution: "9.9.9"
        try:
            compat.install_audio_compat()
        except compat.UnsupportedFasterWhisperVersion as exc:
            message = str(exc)
            assert "expected faster-whisper 1.2.1" in message
            assert "got 9.9.9" in message
        else:
            raise AssertionError("unsupported faster-whisper version was accepted")
        """
    )

    _assert_child_succeeded(result)


def test_install_audio_compat_rejects_preloaded_faster_whisper_audio():
    result = _run_fresh_python(
        """
        import sys
        from types import ModuleType

        from yamabiko.asr import faster_whisper_audio_compat as compat

        sys.modules["faster_whisper.audio"] = ModuleType("faster_whisper.audio")
        try:
            compat.install_audio_compat()
        except RuntimeError as exc:
            message = str(exc)
            assert "already imported before Yamabiko" in message
            assert "faster_whisper.audio" in message
        else:
            raise AssertionError("preloaded faster_whisper.audio was accepted")
        """
    )

    _assert_child_succeeded(result)


def test_production_imports_faster_whisper_only_through_compat_module():
    compat_path = (
        _REPO_ROOT / "yamabiko" / "asr" / "faster_whisper_audio_compat.py"
    ).resolve()
    violations: list[str] = []

    for path in sorted((_REPO_ROOT / "yamabiko").rglob("*.py")):
        if path.resolve() == compat_path:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [
                    alias.name
                    for alias in node.names
                    if alias.name == "faster_whisper"
                    or alias.name.startswith("faster_whisper.")
                ]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported = (
                    [module]
                    if module == "faster_whisper"
                    or module.startswith("faster_whisper.")
                    else []
                )
            else:
                imported = []
            violations.extend(
                f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {name}"
                for name in imported
            )

    assert violations == [], (
        "production must import faster-whisper only through "
        "yamabiko.asr.faster_whisper_audio_compat:\n"
        + "\n".join(violations)
    )
