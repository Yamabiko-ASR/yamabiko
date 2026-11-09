from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.e0_evidence.cli import main
from scripts.e0_evidence.pcm import MasterTransformDag
from scripts.e0_evidence.private_root import PrivateBoundary
from scripts.e0_evidence.registry import master_manifest_to_wire
from scripts.e0_evidence.schema import SCHEMA_VERSION
from scripts.e0_evidence.subtitle_proposals import create_subtitle_proposal
from scripts.e0_evidence.wire import canonical_json_bytes
from tests.e0_evidence.test_replay import _master as replay_master


HEADER = (
    "start_seconds\tend_seconds\tevent_type\tjapanese\tintelligibility\t"
    "speaker\toverlap\tconfidence\tacoustic_tags\tnotes"
)
SOURCE_RATE_HZ = 48_000
SUBTITLE_SRT = (
    "1\r\n"
    "00:01:59,500 --> 00:02:00,500\r\n"
    "<i>冒頭</i>\r\n\r\n"
    "2\r\n"
    "00:02:00,750 --> 00:02:02,000\r\n"
    "今日は\\N\r\n"
    "{\\i1}game night{\\i0}\r\n\r\n"
    "3\r\n"
    "00:06:00,000 --> 00:06:01,000\r\n"
    "outside\r\n"
).encode("utf-8")


def _private_boundary(tmp_path: Path) -> PrivateBoundary:
    root = tmp_path / "private"
    root.mkdir()
    return PrivateBoundary(root, (), ())


def _registered_master(boundary: PrivateBoundary) -> tuple[str, Path]:
    """Register a master whose source-native excerpt is [120 s, 360 s) at 48 kHz."""

    source = boundary.root / "source.private.mkv"
    source.write_bytes(b"synthetic source container")
    synthetic = replay_master(4).manifest
    source_native = replace(
        synthetic.transform_dag.source_native,
        parent_byte_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        input_sample_rate_hz=SOURCE_RATE_HZ,
        excerpt_start_samples=120 * SOURCE_RATE_HZ,
        excerpt_end_samples=360 * SOURCE_RATE_HZ,
    )
    master = replace(
        synthetic,
        sample_count=240 * 16_000,
        duration_samples=240 * 16_000,
        transform_dag=MasterTransformDag(
            source_native=source_native,
            canonical=synthetic.transform_dag.canonical,
        ),
    )
    master_path = boundary.root / "masters" / master.master_id / "master.private.json"
    master_path.parent.mkdir(parents=True)
    master_path.write_bytes(
        canonical_json_bytes(master_manifest_to_wire(master), newline=True)  # type: ignore[arg-type]
    )
    (boundary.root / "locator.private.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "fixtures": {master.fixture_id: {"source": str(source)}},
            },
            newline=True,
        )
    )
    return master.master_id, source


def _fake_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    srt: bytes,
) -> tuple[Path, list[tuple[Path, Path, int]]]:
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_bytes(b"synthetic ffmpeg executable")
    calls: list[tuple[Path, Path, int]] = []

    def extract(executable: Path, source: Path, stream_index: int) -> bytes:
        calls.append((executable, source, stream_index))
        return srt

    monkeypatch.setattr(
        "scripts.e0_evidence.subtitle_proposals._extract_subtitle_stream",
        extract,
    )
    return ffmpeg, calls


def test_create_subtitle_proposal_clips_and_preserves_editorial_cues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = _private_boundary(tmp_path)
    master_id, source = _registered_master(boundary)
    ffmpeg, calls = _fake_ffmpeg(tmp_path, monkeypatch, SUBTITLE_SRT)
    worksheet = boundary.root / "proposal.private.tsv"
    sidecar = boundary.root / "proposal-editorial.private.json"

    first = create_subtitle_proposal(
        master_id,
        1,
        ffmpeg,
        worksheet,
        sidecar,
        boundary,
    )
    first_bytes = (worksheet.read_bytes(), sidecar.read_bytes())

    assert calls == [(ffmpeg, source.resolve(), 1)]
    assert first.cue_count == 2
    assert worksheet.read_text("utf-8").splitlines() == [
        HEADER,
        "0.000\t0.500\tspeech\t冒頭\t\t\t\t\t\t",
        "0.750\t2.000\tspeech\t今日は game night\t\t\t\t\t\t",
    ]
    editorial = json.loads(sidecar.read_text("utf-8"))
    assert editorial == {
        "schema_version": "e0-editorial-subtitle-proposal-2.0",
        "master_id": master_id,
        "subtitle_stream_index": 1,
        "subtitle_sha256": hashlib.sha256(SUBTITLE_SRT).hexdigest(),
        "ffmpeg_sha256": hashlib.sha256(ffmpeg.read_bytes()).hexdigest(),
        "source_sample_rate_hz": SOURCE_RATE_HZ,
        "clip_start_samples": 120 * SOURCE_RATE_HZ,
        "clip_end_samples": 360 * SOURCE_RATE_HZ,
        "cues": [
            {
                "source_ordinal": 1,
                "raw_text": "<i>冒頭</i>",
                "original_start_ms": 119_500,
                "original_end_ms": 120_500,
                "relative_start_ms": 0,
                "relative_end_ms": 500,
            },
            {
                "source_ordinal": 2,
                "raw_text": "今日は\\N\r\n{\\i1}game night{\\i0}",
                "original_start_ms": 120_750,
                "original_end_ms": 122_000,
                "relative_start_ms": 750,
                "relative_end_ms": 2_000,
            },
        ],
    }

    second = create_subtitle_proposal(
        master_id,
        1,
        ffmpeg,
        worksheet,
        sidecar,
        boundary,
    )

    assert second == first
    assert (worksheet.read_bytes(), sidecar.read_bytes()) == first_bytes


def test_annotation_proposal_cli_prints_only_status_and_cue_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    boundary = _private_boundary(tmp_path)
    master_id, source = _registered_master(boundary)
    ffmpeg, _ = _fake_ffmpeg(tmp_path, monkeypatch, SUBTITLE_SRT)
    worksheet = boundary.root / "proposal.private.tsv"
    sidecar = boundary.root / "proposal-editorial.private.json"
    monkeypatch.setattr("scripts.e0_evidence.cli._boundary", lambda args: boundary)

    assert main(
        (
            "annotation-proposal-create",
            "--master-id",
            master_id,
            "--subtitle-stream",
            "1",
            "--ffmpeg",
            str(ffmpeg),
            "--worksheet",
            str(worksheet),
            "--editorial-sidecar",
            str(sidecar),
        )
    ) == 0

    output = capsys.readouterr()
    assert output.err == ""
    assert output.out == "status=created\ncues=2\n"
    assert str(source) not in output.out
    assert "冒頭" not in output.out


def test_srt_with_crlf_trailing_blank_lines_and_an_empty_cue_parses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = _private_boundary(tmp_path)
    master_id, _ = _registered_master(boundary)
    srt = (
        "1\r\n00:02:00,000 --> 00:02:01,000\r\nはい\r\n\r\n"
        "2\r\n00:02:02,000 --> 00:02:03,000\r\n\r\n"
        "3\r\n00:02:04,000 --> 00:02:05,000\r\nうん\r\n\r\n\r\n"
    ).encode("utf-8")
    ffmpeg, _ = _fake_ffmpeg(tmp_path, monkeypatch, srt)
    worksheet = boundary.root / "proposal.private.tsv"
    sidecar = boundary.root / "proposal-editorial.private.json"

    artifacts = create_subtitle_proposal(
        master_id,
        0,
        ffmpeg,
        worksheet,
        sidecar,
        boundary,
    )

    assert artifacts.cue_count == 2
    assert worksheet.read_text("utf-8").splitlines() == [
        HEADER,
        "0.000\t1.000\tspeech\tはい\t\t\t\t\t\t",
        "4.000\t5.000\tspeech\tうん\t\t\t\t\t\t",
    ]
    editorial = json.loads(sidecar.read_text("utf-8"))
    assert [(cue["source_ordinal"], cue["raw_text"]) for cue in editorial["cues"]] == [
        (1, "はい"),
        (3, "うん"),
    ]
