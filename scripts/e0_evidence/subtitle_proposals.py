"""Deterministic, private SRT proposals for subtitle-seeded annotation."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

from ._common import file_sha256, require_sha256, require_uuid4
from .annotations import WORKSHEET_COLUMNS, _write_private_file
from .assets import resolve_fixture_source
from .private_root import PrivateBoundary, require_private_path
from .registry import load_registered_master
from .schema import JsonValue
from .wire import canonical_json_bytes, canonical_json_mapping_from_bytes


EDITORIAL_SIDECAR_SCHEMA_VERSION = "e0-editorial-subtitle-proposal-2.0"
_SIDECAR_FIELDS = frozenset(
    {
        "schema_version",
        "master_id",
        "subtitle_stream_index",
        "subtitle_sha256",
        "ffmpeg_sha256",
        "source_sample_rate_hz",
        "clip_start_samples",
        "clip_end_samples",
        "cues",
    }
)
_CUE_FIELDS = frozenset(
    {
        "source_ordinal",
        "raw_text",
        "original_start_ms",
        "original_end_ms",
        "relative_start_ms",
        "relative_end_ms",
    }
)


_TIMING_RE = re.compile(
    r"^(?P<start>[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}) --> "
    r"(?P<end>[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3})$"
)
_HTML_MARKUP_RE = re.compile(r"<[^>]*>")
_ASS_OVERRIDE_RE = re.compile(r"\{\\[^{}]*\}")
_ASS_LINE_BREAK_RE = re.compile(r"\\[Nn]")
_SRT_LINE_BREAK_RE = re.compile(r"\r\n|\r|\n")
_SRT_BLOCK_BREAK_RE = re.compile(r"\r\n\r\n|\r\r|\n\n")


@dataclass(frozen=True, slots=True)
class SubtitleProposalArtifacts:
    worksheet_path: Path
    editorial_sidecar_path: Path
    cue_count: int


@dataclass(frozen=True, slots=True)
class _Cue:
    source_ordinal: int
    raw_text: str
    original_start_ms: int
    original_end_ms: int


def _timestamp_ms(value: str) -> int:
    hours, minutes, seconds_and_milliseconds = value.split(":", 2)
    seconds, milliseconds = seconds_and_milliseconds.split(",", 1)
    minute_value = int(minutes)
    second_value = int(seconds)
    if minute_value >= 60 or second_value >= 60:
        raise ValueError("subtitle timing is malformed")
    return (
        int(hours) * 3_600_000
        + minute_value * 60_000
        + second_value * 1_000
        + int(milliseconds)
    )


def _parse_cue(block: str, source_ordinal: int) -> _Cue:
    lines = _SRT_LINE_BREAK_RE.split(block, maxsplit=2)
    if len(lines) < 2 or not lines[0].isdigit():
        raise ValueError("subtitle cue is malformed")
    timing = _TIMING_RE.fullmatch(lines[1])
    if timing is None:
        raise ValueError("subtitle timing is malformed")
    start_ms = _timestamp_ms(timing["start"])
    end_ms = _timestamp_ms(timing["end"])
    if end_ms <= start_ms:
        raise ValueError("subtitle timing is malformed")
    # A cue with an index and timing but no text is kept; it is not editable.
    return _Cue(
        source_ordinal=source_ordinal,
        raw_text=lines[2] if len(lines) == 3 else "",
        original_start_ms=start_ms,
        original_end_ms=end_ms,
    )


def _parse_srt(raw: bytes) -> tuple[_Cue, ...]:
    try:
        decoded = raw.decode("utf-8-sig", errors="strict")
    except UnicodeError:
        raise ValueError("subtitle is not valid UTF-8") from None
    # Extra blank lines between or after cues leave blank-only blocks; drop
    # them, and trim the line breaks that remain around each kept block.
    blocks = tuple(
        block.strip("\r\n")
        for block in _SRT_BLOCK_BREAK_RE.split(decoded)
        if block.strip()
    )
    if not blocks:
        raise ValueError("subtitle has no cues")

    cues: list[_Cue] = []
    for source_ordinal, block in enumerate(blocks, start=1):
        try:
            cues.append(_parse_cue(block, source_ordinal))
        except ValueError as exc:
            raise ValueError(f"{exc} (cue={source_ordinal})") from None
    return tuple(cues)


def _clean_editable_text(raw_text: str) -> str:
    without_markup = _HTML_MARKUP_RE.sub("", raw_text)
    without_markup = _ASS_OVERRIDE_RE.sub("", without_markup)
    without_breaks = _ASS_LINE_BREAK_RE.sub(" ", without_markup)
    without_tabs = without_breaks.replace("\t", "")
    return " ".join(without_tabs.split())


def _milliseconds_text(value: int) -> str:
    return f"{value // 1_000}.{value % 1_000:03d}"


def _worksheet_bytes(rows: tuple[tuple[int, int, str], ...]) -> bytes:
    lines = ["\t".join(WORKSHEET_COLUMNS)]
    lines.extend(
        "\t".join(
            (
                _milliseconds_text(start_ms),
                _milliseconds_text(end_ms),
                "speech",
                text,
                "",
                "",
                "",
                "",
                "",
                "",
            )
        )
        for start_ms, end_ms, text in rows
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _existing_bytes_match(path: Path, expected: bytes) -> bool:
    if not path.exists():
        return False
    try:
        return path.read_bytes() == expected
    except OSError:
        raise ValueError("subtitle proposal output is unreadable") from None


def _input_sha256(path: Path, name: str) -> str:
    try:
        return file_sha256(path)
    except OSError:
        raise ValueError(f"{name} is unreadable") from None


def _extract_subtitle_stream(ffmpeg: Path, source: Path, stream_index: int) -> bytes:
    """Return the selected subtitle stream of the source as SRT bytes."""

    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        f"0:s:{stream_index}",
        "-f",
        "srt",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
    except OSError:
        raise ValueError("subtitle stream extraction failed") from None
    if completed.returncode != 0:
        # FFmpeg's stderr names the source path, so it is never rendered.
        raise ValueError("subtitle stream extraction failed")
    return completed.stdout


def _relative_ms(source_ms: int, clip_start_samples: int, rate_hz: int) -> int:
    """Round ``source_ms`` minus the clip start to the nearest millisecond."""

    numerator = source_ms * rate_hz - clip_start_samples * 1_000
    return (2 * numerator + rate_hz) // (2 * rate_hz)


def create_subtitle_proposal(
    master_id: str,
    subtitle_stream_index: int,
    ffmpeg_path: Path,
    worksheet_path: Path,
    editorial_sidecar_path: Path,
    boundary: PrivateBoundary,
) -> SubtitleProposalArtifacts:
    """Extract one master's subtitle stream into proposals and an editorial sidecar.

    The stream is read from the fixture's registered source, and the clip is the
    master's source-native excerpt. The master does not record its FFmpeg
    executable, so the caller names one and the sidecar records its SHA-256.
    """

    if not isinstance(boundary, PrivateBoundary):
        raise TypeError("boundary must be PrivateBoundary")
    if type(subtitle_stream_index) is not int or subtitle_stream_index < 0:
        raise ValueError("subtitle_stream_index must be a non-negative integer")
    worksheet = require_private_path(Path(worksheet_path), boundary)
    sidecar = require_private_path(Path(editorial_sidecar_path), boundary)
    if worksheet == sidecar:
        raise ValueError("subtitle proposal output paths must differ")
    ffmpeg = Path(ffmpeg_path)
    if not ffmpeg.is_absolute() or not ffmpeg.is_file():
        raise ValueError("ffmpeg must be an absolute path to a regular file")
    ffmpeg_sha256 = _input_sha256(ffmpeg, "ffmpeg executable")

    master = load_registered_master(master_id, boundary)
    source_native = master.transform_dag.source_native
    source = resolve_fixture_source(master.fixture_id, master.rights, boundary)
    if _input_sha256(source, "fixture source") != source_native.parent_byte_sha256:
        raise ValueError("fixture source does not match the master's source hash")
    subtitle_bytes = _extract_subtitle_stream(ffmpeg, source, subtitle_stream_index)

    rate_hz = source_native.input_sample_rate_hz
    clip_start = source_native.excerpt_start_samples
    clip_end = source_native.excerpt_end_samples
    # Worksheet rows are millisecond timestamps on the canonical timeline.
    limit_ms = master.sample_count * 1_000 // master.sample_rate_hz
    proposal_rows: list[tuple[int, int, str]] = []
    sidecar_cues: list[dict[str, object]] = []
    for cue in _parse_srt(subtitle_bytes):
        if (
            cue.original_end_ms * rate_hz <= clip_start * 1_000
            or cue.original_start_ms * rate_hz >= clip_end * 1_000
        ):
            continue
        cleaned = _clean_editable_text(cue.raw_text)
        if not cleaned:
            continue
        relative_start_ms = max(
            _relative_ms(cue.original_start_ms, clip_start, rate_hz),
            0,
        )
        relative_end_ms = min(
            _relative_ms(cue.original_end_ms, clip_start, rate_hz),
            limit_ms,
        )
        if relative_end_ms <= relative_start_ms:
            continue
        proposal_rows.append((relative_start_ms, relative_end_ms, cleaned))
        sidecar_cues.append(
            {
                "source_ordinal": cue.source_ordinal,
                "raw_text": cue.raw_text,
                "original_start_ms": cue.original_start_ms,
                "original_end_ms": cue.original_end_ms,
                "relative_start_ms": relative_start_ms,
                "relative_end_ms": relative_end_ms,
            }
        )
    if not proposal_rows:
        raise ValueError("subtitle proposal has no editable cues")

    worksheet_bytes = _worksheet_bytes(tuple(proposal_rows))
    sidecar_payload: dict[str, object] = {
        "schema_version": EDITORIAL_SIDECAR_SCHEMA_VERSION,
        "master_id": master.master_id,
        "subtitle_stream_index": subtitle_stream_index,
        "subtitle_sha256": hashlib.sha256(subtitle_bytes).hexdigest(),
        "ffmpeg_sha256": ffmpeg_sha256,
        "source_sample_rate_hz": rate_hz,
        "clip_start_samples": clip_start,
        "clip_end_samples": clip_end,
        "cues": sidecar_cues,
    }
    sidecar_bytes = canonical_json_bytes(
        cast(Mapping[str, JsonValue], sidecar_payload),
        newline=True,
    )

    worksheet_matches = _existing_bytes_match(worksheet, worksheet_bytes)
    sidecar_matches = _existing_bytes_match(sidecar, sidecar_bytes)
    if (worksheet.exists() and not worksheet_matches) or (
        sidecar.exists() and not sidecar_matches
    ):
        raise ValueError("conflicting subtitle proposal output reuse")
    if not worksheet_matches:
        _write_private_file(worksheet, worksheet_bytes)
    if not sidecar_matches:
        _write_private_file(sidecar, sidecar_bytes)
    return SubtitleProposalArtifacts(worksheet, sidecar, len(proposal_rows))


def _sidecar_int(value: object, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("editorial subtitle sidecar is invalid")
    return value


def load_editorial_sidecar(raw: bytes) -> dict[str, object]:
    """Validate the exact bytes of one editorial sidecar written above."""

    try:
        payload = canonical_json_mapping_from_bytes(raw, "editorial subtitle sidecar")
    except (TypeError, ValueError):
        raise ValueError("editorial subtitle sidecar is invalid") from None
    if (
        set(payload) != _SIDECAR_FIELDS
        or payload["schema_version"] != EDITORIAL_SIDECAR_SCHEMA_VERSION
    ):
        raise ValueError("editorial subtitle sidecar is invalid")
    require_uuid4(payload["master_id"], "editorial subtitle master_id")
    _sidecar_int(payload["subtitle_stream_index"], minimum=0)
    for field_name in ("subtitle_sha256", "ffmpeg_sha256"):
        try:
            require_sha256(payload[field_name], field_name)
        except (TypeError, ValueError):
            raise ValueError("editorial subtitle sidecar is invalid") from None
    _sidecar_int(payload["source_sample_rate_hz"], minimum=1)
    clip_start = _sidecar_int(payload["clip_start_samples"], minimum=0)
    _sidecar_int(payload["clip_end_samples"], minimum=clip_start + 1)
    cues = payload["cues"]
    if not isinstance(cues, list) or not cues:
        raise ValueError("editorial subtitle sidecar is invalid")
    for cue in cues:
        if not isinstance(cue, dict) or set(cue) != _CUE_FIELDS:
            raise ValueError("editorial subtitle sidecar is invalid")
        if type(cue["raw_text"]) is not str or not cue["raw_text"]:
            raise ValueError("editorial subtitle sidecar is invalid")
        _sidecar_int(cue["source_ordinal"], minimum=1)
        original_start = _sidecar_int(cue["original_start_ms"], minimum=0)
        _sidecar_int(cue["original_end_ms"], minimum=original_start + 1)
        relative_start = _sidecar_int(cue["relative_start_ms"], minimum=0)
        _sidecar_int(cue["relative_end_ms"], minimum=relative_start + 1)
    return payload


__all__ = [
    "EDITORIAL_SIDECAR_SCHEMA_VERSION",
    "SubtitleProposalArtifacts",
    "create_subtitle_proposal",
    "load_editorial_sidecar",
]
