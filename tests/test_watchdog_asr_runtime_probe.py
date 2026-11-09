"""Unit contracts for the frozen-only ASR runtime release smoke."""
from __future__ import annotations

import sys

import pytest

from yamabiko import watchdog
from yamabiko.asr import faster_whisper_audio_compat as compat


_EXPECTED_PUBLIC_HELP = """yamabiko - Japanese audio -> subtitle pipeline

Usage:
    yamabiko                     Normal launch (watchdog supervises
                                 pipeline + overlay + llama-server).
    yamabiko --version           Print version and exit.
    yamabiko --help              Show this message and exit.
    yamabiko --max-segments N    Dev: exit the pipeline after N
                                 translated segments (0 = run forever).
                                 Forwarded to yamabiko.main.
    yamabiko --config PATH       Dev: load config from PATH instead of
                                 the layered default_config.yaml +
                                 user_overrides.yaml. Forwarded to
                                 yamabiko.main.

Internal dispatch (not typically user-invoked):
    yamabiko --run-main          Drive the pipeline in this process
                                 (watchdog subprocess re-entry).
    yamabiko --run-overlay       Drive the Qt desktop overlay in this
                                 process (watchdog subprocess re-entry).

See docs/RELEASE.md for the release workflow and the main-*.log files
in %LOCALAPPDATA%\\Yamabiko\\logs for runtime logs (one per process:
watchdog, pipeline, overlay).
"""


def test_public_help_is_byte_for_byte_unchanged_and_probe_is_unlisted(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(sys, "argv", ["yamabiko", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        watchdog.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert output == _EXPECTED_PUBLIC_HELP + "\n"
    assert "--verify-asr-runtime" not in output


def test_internal_asr_probe_route_precedes_public_help(monkeypatch, capsys):
    """The suppressed flag must win before the public ``--help`` branch.

    Seeded defect: move the internal route below public help; the sentinel
    marker is replaced by the help snapshot and this fails.
    """
    def _fake_probe() -> None:
        print("PROBE_ROUTE_OK")
        raise SystemExit(0)

    monkeypatch.setattr(
        watchdog,
        "_verify_asr_runtime_and_exit",
        _fake_probe,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["yamabiko", "--verify-asr-runtime", "--help"],
    )

    with pytest.raises(SystemExit) as exc_info:
        watchdog.main()

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == "PROBE_ROUTE_OK\n"


def test_asr_runtime_probe_reports_metadata_and_loads_compat_seam(
    monkeypatch,
    capsys,
):
    """Probe the exact operations without constructing ``WhisperModel``."""
    requested_distributions: list[str] = []

    def _version(distribution: str) -> str:
        requested_distributions.append(distribution)
        return "1.2.1"

    class WhisperModel:
        def __init__(self, *args, **kwargs):
            raise AssertionError("release smoke must not instantiate the model")

    monkeypatch.setattr(compat.metadata, "version", _version)
    monkeypatch.setattr(compat, "import_whisper_model", lambda: WhisperModel)

    with pytest.raises(SystemExit) as exc_info:
        watchdog._verify_asr_runtime_and_exit()

    assert exc_info.value.code == 0
    assert requested_distributions == ["faster-whisper"]
    assert capsys.readouterr().out == (
        "YAMABIKO_ASR_RUNTIME_OK faster-whisper=1.2.1\n"
    )
