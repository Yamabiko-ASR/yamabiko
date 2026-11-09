"""Append-only, verifiable evidence ledger for E0 runs."""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    BinaryIO,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
)

from scripts.e0_evidence._common import (
    atomic_write_bytes,
    require_sha256,
    require_uuid4,
)
from scripts.e0_evidence.schema import (
    SCHEMA_VERSION,
    ArmKind,
    ArmStatus,
    ArmTerminalStatus,
    DecoderAdapter,
    DiagnosticCode,
    EndpointEmitReason,
    EvidenceRecord,
    EvidenceRecordDraft,
    IntegrityFailureCode,
    IntegrityReport,
    LossCode,
    ObservationStatus,
    PrivateContentClass,
    PrivateTextRecord,
    RecordKind,
    ReplayStopKind,
    RunFinalManifest,
    RunStartManifest,
    RunStatus,
    SampleSpan,
    Stage,
    StreamName,
    StreamSummary,
    TerminalOutcome,
    TextRole,
    new_opaque_id,
)
from scripts.e0_evidence.pcm import capture_replay_provenance_sha256
from scripts.e0_evidence.private_root import PrivateBoundary, require_private_path
from scripts.e0_evidence.wire import (
    WireValidationError,
    canonical_json_bytes,
    canonical_json_mapping_from_bytes,
    canonical_sha256,
    evidence_record_from_wire,
    evidence_record_to_wire,
    inherited_classification,
    private_text_record_from_wire,
    private_text_record_to_wire,
    run_final_from_wire,
    run_final_to_wire,
    run_start_from_wire,
    run_start_to_wire,
)
from yamabiko.asr.asr_client import (
    _COMPRESSION_RATIO_THRESHOLD,
    _MIN_COMPRESSION_GATE_LEN,
)
from yamabiko.asr.faster_whisper_audio_compat import import_whisper_model
from yamabiko.models import SILERO_SHA256


_RUN_START_NAME = "run-start.json"
_CANONICAL_NAME = "ledger.jsonl"
_PRIVATE_TEXT_NAME = "private-text.jsonl"
_RUN_FINAL_NAME = "run-final.json"
_TERMINAL_ARM_STATUSES = {
    ArmStatus.COMPLETE,
    ArmStatus.INCOMPLETE,
    ArmStatus.FAILED,
    ArmStatus.CANCELLED,
    ArmStatus.NOT_RUN,
}
_DIAGNOSTIC_DECISIONS = frozenset(item.value for item in DiagnosticCode)
_ENDPOINT_DECISIONS = frozenset(item.value for item in EndpointEmitReason)
_TERMINAL_OUTCOMES = frozenset(item.value for item in TerminalOutcome)
_DECISION_VALUES_BY_KIND = {
    RecordKind.RUN_STATUS: frozenset(
        item.value for item in RunStatus if item is not RunStatus.RUNNING
    ),
    RecordKind.ARM_STATUS: frozenset(item.value for item in _TERMINAL_ARM_STATUSES),
    RecordKind.GATE_TRANSITION: _ENDPOINT_DECISIONS,
    RecordKind.WINDOW: _ENDPOINT_DECISIONS,
    RecordKind.DECODER_CALL: frozenset(
        {
            DiagnosticCode.DECODER_NO_YIELD.value,
            DiagnosticCode.UNKNOWN_DECODER_GATE.value,
        }
    ),
    RecordKind.WRAPPER_DECISION: frozenset(
        {
            DiagnosticCode.WRAPPER_CONFIDENCE_GATE.value,
            DiagnosticCode.WRAPPER_COMPRESSION_REJECT.value,
            DiagnosticCode.WRAPPER_EMPTY.value,
            TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value,
        }
    ),
    RecordKind.DIAGNOSTIC: _DIAGNOSTIC_DECISIONS,
    RecordKind.EVENT_FATE: _TERMINAL_OUTCOMES,
    RecordKind.INTEGRITY: frozenset(item.value for item in IntegrityFailureCode),
    RecordKind.DISCONTINUITY: frozenset(
        {
            DiagnosticCode.CAPTURE_DISCONTINUITY.value,
            DiagnosticCode.CALLBACK_OVERFLOW.value,
            DiagnosticCode.PAUSED_DISCARD.value,
        }
    ),
}
_ARM_REASON_VALUES = frozenset(
    {
        *(item.value for item in DiagnosticCode),
        *(item.value for item in IntegrityFailureCode),
        *(item.value for item in LossCode),
        *(item.value for item in ObservationStatus),
        *(item.value for item in TerminalOutcome),
    }
)
_TEXT_ROLE_BY_KIND = {
    RecordKind.RAW_HYPOTHESIS: TextRole.RAW_HYPOTHESIS,
    RecordKind.WRAPPER_DECISION: TextRole.WRAPPER_HYPOTHESIS,
}
_MEASUREMENT_UNITS = {
    "wall_time_seconds": "seconds",
    "rtf": "ratio",
    "cold_start_seconds": "seconds",
    "peak_rss_bytes": "bytes",
    "peak_vram_bytes": "bytes",
}
_VRAM_SCOPE = "device_delta"


class EvidenceSink(Protocol):
    def emit(self, draft: EvidenceRecordDraft) -> EvidenceRecord: ...


class IncompleteRunError(ValueError):
    def __init__(self, report: IntegrityReport) -> None:
        self.report = report
        codes = ",".join(reason.value for reason in report.reasons)
        super().__init__(f"evidence run is incomplete: {codes or 'unknown'}")


def _unique_reasons(
    reasons: Iterable[IntegrityFailureCode],
) -> tuple[IntegrityFailureCode, ...]:
    result: list[IntegrityFailureCode] = []
    for reason in reasons:
        if reason not in result:
            result.append(reason)
    return tuple(result)


def _require_plain_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field_name} must be an integer of at least {minimum}")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


_GATE_STATE_KEYS = {
    "in_speech",
    "retained_frames",
    "speech_frames",
    "consecutive_silence_frames",
    "silence_since_speech_frames",
    "frames_since_open",
}


def _validate_gate_state(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _GATE_STATE_KEYS:
        raise ValueError(f"{field_name} has an invalid shape")
    in_speech = _require_bool(value["in_speech"], f"{field_name}.in_speech")
    for key in _GATE_STATE_KEYS - {"in_speech"}:
        _require_plain_int(value[key], f"{field_name}.{key}")
    if not in_speech and any(
        value[key] != 0
        for key in (
            "speech_frames",
            "consecutive_silence_frames",
            "silence_since_speech_frames",
            "frames_since_open",
        )
    ):
        raise ValueError(f"{field_name} closed-gate counters must be zero")
    return value


def _require_probability(value: object, field_name: str) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        raise ValueError(f"{field_name} must be a finite probability")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field_name} must be a finite probability")
    return probability


def _require_finite_number(value: object, field_name: str) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


_REPLAY_CONFIG_KEYS = {
    "sample_rate",
    "min_chunk_ms",
    "max_chunk_ms",
    "min_silence_ms",
    "speech_pad_ms",
    "glue_max_silence_ms",
    "attach_max_ms",
    "drop_floor_prob",
    "drop_floor_ms",
    "dip_split_enabled",
    "dip_split_threshold",
    "dip_split_min_piece_ms",
}
_REPLAY_CONTRACT_KEYS = {
    "master_pcm_sha256",
    "replay_pcm_sha256",
    "capture_replay_sha256",
    "model_sha256",
    "probe_source_sha256",
    "silero_vad_source_sha256",
    "probe_contract_version",
    "effective_config_sha256",
    "frame_samples",
    "threshold",
    "negative_threshold",
    "timeline_id",
    "master_sample_count",
    "complete_frame_count",
    "partial_frame_samples",
    "stop_kind",
    "config",
    "vad_assist",
}
_VAD_ASSIST_KEYS = {
    "model_sha256",
    "sound_tagger_source_sha256",
    "silero_chunker_source_sha256",
    "layout",
}
_FRAME_KEYS = {
    "frame_ordinal",
    "vad_probability",
    "silero_probability",
    "sound_tag_term",
    "gate_in_speech_after",
}
# ``run-start.json`` declares, under ``approved_contracts``, the source and
# dependency contracts its recorders enforced (``scripts.e0_evidence.contracts``).
# Records are verified against that declaration, never the current module.
_APPROVED_CONTRACT_SHA256_KEYS = (
    "decoder_signature_sha256",
    "frame_gate_source_sha256",
    "silero_chunker_source_sha256",
    "silero_vad_source_sha256",
    "sound_tag_assist_layout_sha256",
    "sound_tagger_model_sha256",
    "sound_tagger_source_sha256",
    "wrapper_source_sha256",
)
_APPROVED_CONTRACT_VERSION_KEYS = ("faster_whisper_version", "yamabiko_version_pin")
_APPROVED_CONTRACT_INT_KEYS = ("decoder_contract_version", "probe_contract_version")
_APPROVED_CONTRACT_KEYS = frozenset(
    {
        *_APPROVED_CONTRACT_SHA256_KEYS,
        *_APPROVED_CONTRACT_VERSION_KEYS,
        *_APPROVED_CONTRACT_INT_KEYS,
        "decoder_random_seed",
    }
)
_VAD_ARTIFACT_KEYS = {"sha256", "byte_count", "production_pin"}
_REPLAY_TERMINAL_KEYS = {
    "replay_contract_id",
    "stop_kind",
    "window_count",
    "gate_transition_count",
    "pending_span_count",
    "has_partial_frame",
    "diagnostic_codes",
}
_DECODER_CALL_KEYS = {
    "call_mode",
    "changed_argument_keys",
    "decoder_contract_version",
    "decoder_input_pcm_sha256",
    "decoder_input_samples",
    "decoder_random_seed",
    "delegate_class_sha256",
    "delegate_instance_id",
    "delegate_signature_sha256",
    "dependency_version",
    "effective_arguments",
    "effective_arguments_sha256",
    "mapping_version",
    "model_artifact_sha256",
    "oracle_component_id",
    "oracle_component_map_sha256",
    "oracle_event_ids",
    "oracle_reference_set_id",
    "oracle_reference_set_sha256",
    "oracle_slice_pcm_sha256",
    "parent_call_record_id",
    "raw_segment_count",
    "sample_rate",
    "source_artifact_id",
    "source_pcm_sha256",
    "source_to_decoder_map",
    "source_to_decoder_map_id",
    "source_to_decoder_map_sha256",
    "synthetic_left_pad_samples",
    "synthetic_right_pad_samples",
    "wrapper_source_sha256",
}


_RAW_HYPOTHESIS_KEYS = {
    "avg_logprob",
    "compression_ratio",
    "end_decoder_sample",
    "no_speech_prob",
    "raw_end_seconds",
    "raw_start_seconds",
    "segment_ordinal",
    "start_decoder_sample",
    "synthetic_anchor_sample",
    "synthetic_anchor_timeline_id",
    "synthetic_pad_hypothesis",
    "temperature",
    "text_id",
    "timestamp_clamped",
}
_PINNED_DECODER_ARGUMENT_NAMES = {
    "language",
    "task",
    "log_progress",
    "beam_size",
    "best_of",
    "patience",
    "length_penalty",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "temperature",
    "compression_ratio_threshold",
    "log_prob_threshold",
    "no_speech_threshold",
    "condition_on_previous_text",
    "prompt_reset_on_temperature",
    "initial_prompt",
    "prefix",
    "suppress_blank",
    "suppress_tokens",
    "without_timestamps",
    "max_initial_timestamp",
    "word_timestamps",
    "prepend_punctuations",
    "append_punctuations",
    "multilingual",
    "vad_filter",
    "vad_parameters",
    "max_new_tokens",
    "chunk_length",
    "clip_timestamps",
    "hallucination_silence_threshold",
    "hotwords",
    "language_detection_threshold",
    "language_detection_segments",
}
# The onnx-asr NeMo TDT candidate (``scripts.e0_evidence.nemo_tdt``) shares
# the source map, raw projection, and wrapper-record contract above. It has no
# Whisper argument map, confidence statistics, or temperature fallback, so its
# records name the adapter and carry adapter facts under ``nemo_tdt``.
_NEMO_TDT = DecoderAdapter.ONNX_ASR_NEMO_TDT.value
_NEMO_TDT_DECODER_CALL_KEYS = {
    "call_mode",
    "decoder_adapter",
    "decoder_contract_version",
    "decoder_input_pcm_sha256",
    "decoder_input_samples",
    "mapping_version",
    "model_artifact_sha256",
    "nemo_tdt",
    "oracle_component_id",
    "oracle_component_map_sha256",
    "oracle_event_ids",
    "oracle_reference_set_id",
    "oracle_reference_set_sha256",
    "oracle_slice_pcm_sha256",
    "parent_call_record_id",
    "raw_segment_count",
    "sample_rate",
    "source_artifact_id",
    "source_pcm_sha256",
    "source_to_decoder_map",
    "source_to_decoder_map_id",
    "source_to_decoder_map_sha256",
    "synthetic_left_pad_samples",
    "synthetic_right_pad_samples",
}
_NEMO_TDT_CALL_FACT_KEYS = {"model_type", "timestamp_frame_seconds", "token_count"}
_NEMO_TDT_RAW_HYPOTHESIS_KEYS = {
    "end_decoder_sample",
    "nemo_tdt",
    "raw_end_seconds",
    "raw_start_seconds",
    "segment_ordinal",
    "start_decoder_sample",
    "synthetic_anchor_sample",
    "synthetic_anchor_timeline_id",
    "synthetic_pad_hypothesis",
    "text_id",
    "timestamp_clamped",
}
_NEMO_TDT_RAW_FACT_KEYS = {"compression_ratio", "last_text_token_seconds"}
_NEMO_TDT_MODEL_TYPE = "nemo-conformer-tdt"
_NEMO_TDT_ONNX_ASR_VERSION = "0.12.0"
_NEMO_TDT_EXECUTION_PROVIDER = "CPUExecutionProvider"
_NEMO_TDT_IDENTITY_KEYS = {
    "name",
    "model_type",
    "quantization",
    "onnx_asr_version",
    "onnxruntime_version",
    "execution_provider",
    "model_artifact_manifest",
}
_NEMO_TDT_FP32_FILES = frozenset(
    {
        "config.json",
        "decoder_joint-model.onnx",
        "decoder_joint-model.onnx_data",
        "encoder-model.onnx",
        "encoder-model.onnx_data",
        "vocab.txt",
    }
)


def _approved_contracts(start: RunStartManifest) -> Mapping[str, object] | None:
    """Return the run's declared approved contracts, or None when malformed."""

    approved = start.details.get("approved_contracts")
    if not isinstance(approved, Mapping) or set(approved) != _APPROVED_CONTRACT_KEYS:
        return None
    try:
        for key in _APPROVED_CONTRACT_SHA256_KEYS:
            require_sha256(approved[key], f"approved_contracts.{key}")
        for key in _APPROVED_CONTRACT_VERSION_KEYS:
            if type(approved[key]) is not str or not approved[key]:
                raise ValueError(f"approved_contracts.{key} must be a version string")
        for key in _APPROVED_CONTRACT_INT_KEYS:
            _require_plain_int(approved[key], f"approved_contracts.{key}", minimum=1)
        _require_plain_int(
            approved["decoder_random_seed"],
            "approved_contracts.decoder_random_seed",
        )
    except (TypeError, ValueError):
        return None
    return approved


def _require_approved(
    approved: Mapping[str, object] | None,
) -> Mapping[str, object]:
    if approved is None:
        raise ValueError("the run start declares no valid approved_contracts")
    return approved


def _validate_replay_contract_details(
    details: Mapping[str, object],
    approved: Mapping[str, object] | None,
) -> None:
    if set(details) != _REPLAY_CONTRACT_KEYS:
        raise ValueError("integrity replay-contract details have an invalid shape")
    for key in (
        "master_pcm_sha256",
        "replay_pcm_sha256",
        "capture_replay_sha256",
        "model_sha256",
        "probe_source_sha256",
        "silero_vad_source_sha256",
        "effective_config_sha256",
    ):
        require_sha256(details[key], f"details.{key}")
    approved = _require_approved(approved)
    probe_version = _require_plain_int(
        details["probe_contract_version"],
        "details.probe_contract_version",
        minimum=1,
    )
    if probe_version != approved["probe_contract_version"]:
        raise ValueError(
            "details.probe_contract_version is not the run's approved contract"
        )
    if details["probe_source_sha256"] != approved["frame_gate_source_sha256"]:
        raise ValueError("details.probe_source_sha256 is not the run's approved contract")
    if details["silero_vad_source_sha256"] != approved["silero_vad_source_sha256"]:
        raise ValueError(
            "details.silero_vad_source_sha256 is not the run's approved contract"
        )
    frame_samples = _require_plain_int(
        details["frame_samples"],
        "details.frame_samples",
        minimum=1,
    )
    if frame_samples != 512:
        raise ValueError("details.frame_samples must be 512")
    require_uuid4(details["timeline_id"], "details.timeline_id")
    master_sample_count = _require_plain_int(
        details["master_sample_count"],
        "details.master_sample_count",
        minimum=1,
    )
    complete_frame_count = _require_plain_int(
        details["complete_frame_count"],
        "details.complete_frame_count",
    )
    partial_frame_samples = _require_plain_int(
        details["partial_frame_samples"],
        "details.partial_frame_samples",
    )
    if (
        partial_frame_samples >= frame_samples
        or complete_frame_count * frame_samples + partial_frame_samples
        != master_sample_count
    ):
        raise ValueError("replay frame counts do not match the master timeline")
    try:
        ReplayStopKind(details["stop_kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("details.stop_kind is invalid") from exc
    threshold = _require_probability(details["threshold"], "details.threshold")
    negative_threshold = _require_probability(
        details["negative_threshold"],
        "details.negative_threshold",
    )
    if negative_threshold != max(threshold - 0.15, 0.01):
        raise ValueError("details.negative_threshold does not match production")
    config = details["config"]
    if not isinstance(config, Mapping) or set(config) != _REPLAY_CONFIG_KEYS:
        raise ValueError("details.config has an invalid shape")
    for key in (
        "sample_rate",
        "min_chunk_ms",
        "max_chunk_ms",
        "min_silence_ms",
        "speech_pad_ms",
        "glue_max_silence_ms",
        "attach_max_ms",
        "drop_floor_ms",
        "dip_split_min_piece_ms",
    ):
        _require_plain_int(config[key], f"details.config.{key}")
    if config["sample_rate"] != 16_000:
        raise ValueError("details.config.sample_rate must be 16000")
    _require_probability(config["drop_floor_prob"], "details.config.drop_floor_prob")
    _require_probability(
        config["dip_split_threshold"],
        "details.config.dip_split_threshold",
    )
    _require_bool(config["dip_split_enabled"], "details.config.dip_split_enabled")
    effective_mapping = {
        "config": dict(config),
        "frame_samples": 512,
        "threshold": threshold,
    }
    if canonical_sha256(effective_mapping) != details["effective_config_sha256"]:
        raise ValueError("details.effective_config_sha256 does not match config")
    assist = details["vad_assist"]
    if not isinstance(assist, Mapping) or set(assist) != _VAD_ASSIST_KEYS:
        raise ValueError("details.vad_assist has an invalid shape")
    for key, approved_key in (
        ("model_sha256", "sound_tagger_model_sha256"),
        ("sound_tagger_source_sha256", "sound_tagger_source_sha256"),
        ("silero_chunker_source_sha256", "silero_chunker_source_sha256"),
    ):
        if assist[key] != approved[approved_key]:
            raise ValueError(
                f"details.vad_assist.{key} is not the run's approved contract"
            )
    if (
        not isinstance(assist["layout"], Mapping)
        or canonical_sha256(assist["layout"])
        != approved["sound_tag_assist_layout_sha256"]
    ):
        raise ValueError("details.vad_assist.layout is not the run's approved contract")


def _validate_replay_terminal_details(details: Mapping[str, object]) -> None:
    if set(details) != _REPLAY_TERMINAL_KEYS:
        raise ValueError("integrity replay-terminal details have an invalid shape")
    require_uuid4(details["replay_contract_id"], "details.replay_contract_id")
    try:
        ReplayStopKind(details["stop_kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("details.stop_kind is invalid") from exc
    _require_plain_int(details["window_count"], "details.window_count")
    _require_plain_int(
        details["gate_transition_count"],
        "details.gate_transition_count",
    )
    _require_plain_int(
        details["pending_span_count"],
        "details.pending_span_count",
    )
    _require_bool(details["has_partial_frame"], "details.has_partial_frame")
    diagnostic_codes = details["diagnostic_codes"]
    if not isinstance(diagnostic_codes, tuple):
        raise ValueError("details.diagnostic_codes must be an array")
    codes = diagnostic_codes
    for code in codes:
        try:
            DiagnosticCode(code)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "details.diagnostic_codes contains an invalid code"
            ) from exc
    if len(codes) != len(set(codes)):
        raise ValueError("details.diagnostic_codes must not contain duplicates")


def _validate_safe_argument(value: object, field_name: str) -> None:
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        _require_finite_number(value, field_name)
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_safe_argument(item, f"{field_name}[{index}]")
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} has an unsupported value")
    if set(value) == {"kind", "sha256", "utf8_bytes"}:
        if value["kind"] != "private_utf8":
            raise ValueError(f"{field_name}.kind is invalid")
        require_sha256(value["sha256"], f"{field_name}.sha256")
        _require_plain_int(value["utf8_bytes"], f"{field_name}.utf8_bytes")
        return
    for key, item in value.items():
        if type(key) is not str or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        _validate_safe_argument(item, f"{field_name}.{key}")


def _validate_decoder_map(
    details: Mapping[str, object],
) -> tuple[int, int, int]:
    intervals = details["source_to_decoder_map"]
    if not isinstance(intervals, tuple) or not intervals:
        raise ValueError("details.source_to_decoder_map must be a non-empty array")
    cursor = 0
    real_samples = 0
    for index, interval in enumerate(intervals):
        if not isinstance(interval, Mapping) or set(interval) != {
            "decoder_end",
            "decoder_start",
            "source_span",
        }:
            raise ValueError("details.source_to_decoder_map interval is invalid")
        start = _require_plain_int(
            interval["decoder_start"],
            "details.source_to_decoder_map.decoder_start",
        )
        end = _require_plain_int(
            interval["decoder_end"],
            "details.source_to_decoder_map.decoder_end",
            minimum=1,
        )
        if start != cursor or end <= start:
            raise ValueError("details.source_to_decoder_map must be contiguous")
        source = interval["source_span"]
        if source is None:
            if index not in (0, len(intervals) - 1):
                raise ValueError("synthetic intervals are allowed only at boundaries")
        else:
            if not isinstance(source, Mapping) or set(source) != {
                "end",
                "start",
                "timeline_id",
            }:
                raise ValueError("details.source_to_decoder_map source span is invalid")
            require_uuid4(
                source["timeline_id"],
                "details.source_to_decoder_map.timeline_id",
            )
            source_start = _require_plain_int(
                source["start"],
                "details.source_to_decoder_map.source_start",
            )
            source_end = _require_plain_int(
                source["end"],
                "details.source_to_decoder_map.source_end",
                minimum=1,
            )
            if source_end <= source_start or source_end - source_start != end - start:
                raise ValueError("source and decoder interval lengths must agree")
            real_samples += end - start
        cursor = end
    decoder_samples = _require_plain_int(
        details["decoder_input_samples"],
        "details.decoder_input_samples",
        minimum=1,
    )
    if cursor != decoder_samples or not real_samples:
        raise ValueError("source map does not account for decoder input")
    left_pad = _require_plain_int(
        details["synthetic_left_pad_samples"],
        "details.synthetic_left_pad_samples",
    )
    right_pad = _require_plain_int(
        details["synthetic_right_pad_samples"],
        "details.synthetic_right_pad_samples",
    )
    actual_left = (
        intervals[0]["decoder_end"] - intervals[0]["decoder_start"]
        if intervals[0]["source_span"] is None
        else 0
    )
    actual_right = (
        intervals[-1]["decoder_end"] - intervals[-1]["decoder_start"]
        if intervals[-1]["source_span"] is None
        else 0
    )
    if (
        left_pad != actual_left
        or right_pad != actual_right
        or left_pad != right_pad
        or real_samples + left_pad + right_pad != decoder_samples
    ):
        raise ValueError("synthetic padding does not match the source map")
    expected_map_sha256 = canonical_sha256(
        {
            "decoder_samples": decoder_samples,
            "intervals": intervals,
            "mapping_version": details["mapping_version"],
            "sample_rate": details["sample_rate"],
        }
    )
    if expected_map_sha256 != details["source_to_decoder_map_sha256"]:
        raise ValueError("details.source_to_decoder_map_sha256 is invalid")
    return decoder_samples, left_pad, right_pad


def _validate_decoder_call_details(
    decision: str | None,
    details: Mapping[str, object],
    approved: Mapping[str, object] | None,
) -> None:
    if set(details) != _DECODER_CALL_KEYS:
        raise ValueError("decoder call details have an invalid shape")
    try:
        call_mode = ArmKind(details["call_mode"])
    except (TypeError, ValueError) as exc:
        raise ValueError("details.call_mode is invalid") from exc
    approved = _require_approved(approved)
    if details["dependency_version"] != approved["faster_whisper_version"]:
        raise ValueError("details.dependency_version is not the run's approved contract")
    if (
        details["decoder_contract_version"] != approved["decoder_contract_version"]
        or details["decoder_random_seed"] != approved["decoder_random_seed"]
        or details["mapping_version"] != 1
        or details["sample_rate"] != 16_000
    ):
        raise ValueError(
            "decoder call contract is not the run's approved contract at 16000 Hz"
        )
    for key in (
        "decoder_input_pcm_sha256",
        "delegate_class_sha256",
        "delegate_signature_sha256",
        "effective_arguments_sha256",
        "model_artifact_sha256",
        "source_pcm_sha256",
        "source_to_decoder_map_sha256",
        "wrapper_source_sha256",
    ):
        require_sha256(details[key], f"details.{key}")
    for key in (
        "delegate_instance_id",
        "source_artifact_id",
        "source_to_decoder_map_id",
    ):
        require_uuid4(details[key], f"details.{key}")
    if details["delegate_signature_sha256"] != approved["decoder_signature_sha256"]:
        raise ValueError(
            "details.delegate_signature_sha256 is not the run's approved contract"
        )
    if details["wrapper_source_sha256"] != approved["wrapper_source_sha256"]:
        raise ValueError("details.wrapper_source_sha256 is not the run's approved contract")
    _validate_decoder_map(details)
    raw_count = _require_plain_int(
        details["raw_segment_count"],
        "details.raw_segment_count",
    )
    if (raw_count == 0) != (decision == DiagnosticCode.DECODER_NO_YIELD.value):
        raise ValueError("decoder no-yield decision and raw count do not agree")
    effective = details["effective_arguments"]
    if (
        not isinstance(effective, Mapping)
        or set(effective) != _PINNED_DECODER_ARGUMENT_NAMES
    ):
        raise ValueError("details.effective_arguments is not the pinned full map")
    for name, value in effective.items():
        _validate_safe_argument(value, f"details.effective_arguments.{name}")
    if canonical_sha256(effective) != details["effective_arguments_sha256"]:
        raise ValueError("details.effective_arguments_sha256 is invalid")
    parent_call_id = details["parent_call_record_id"]
    changed = details["changed_argument_keys"]
    if not isinstance(changed, tuple) or any(type(key) is not str for key in changed):
        raise ValueError("details.changed_argument_keys must be a string array")
    if len(changed) != len(set(changed)):
        raise ValueError("details.changed_argument_keys contains duplicates")
    if call_mode in (ArmKind.P0, ArmKind.P1):
        if parent_call_id is not None or changed:
            raise ValueError("wrapper arms cannot declare a parent call or change")
    else:
        require_uuid4(parent_call_id, "details.parent_call_record_id")
        expected_changes = ("no_speech_threshold",) if call_mode is ArmKind.P2B else ()
        if changed != expected_changes:
            raise ValueError("direct arm changed arguments are invalid")
    expected_threshold = None if call_mode is ArmKind.P2B else 0.6
    if effective["no_speech_threshold"] != expected_threshold:
        raise ValueError("effective no-speech threshold does not match the arm")
    _validate_decoder_oracle_provenance(call_mode, details)


def _validate_decoder_oracle_provenance(
    call_mode: ArmKind,
    details: Mapping[str, object],
) -> None:
    oracle_values = (
        details["oracle_component_id"],
        details["oracle_reference_set_id"],
        details["oracle_reference_set_sha256"],
        details["oracle_component_map_sha256"],
        details["oracle_slice_pcm_sha256"],
    )
    if call_mode is ArmKind.P1:
        if not all(value is not None for value in oracle_values):
            raise ValueError("P1 requires complete oracle provenance")
        require_uuid4(oracle_values[0], "details.oracle_component_id")
        require_uuid4(oracle_values[1], "details.oracle_reference_set_id")
        require_sha256(oracle_values[2], "details.oracle_reference_set_sha256")
        require_sha256(oracle_values[3], "details.oracle_component_map_sha256")
        require_sha256(oracle_values[4], "details.oracle_slice_pcm_sha256")
        event_ids = details["oracle_event_ids"]
        if not isinstance(event_ids, tuple) or not event_ids:
            raise ValueError("P1 requires oracle event IDs")
        for event_id in event_ids:
            require_uuid4(event_id, "details.oracle_event_ids item")
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("details.oracle_event_ids contains duplicates")
        if (
            details["source_artifact_id"] != details["oracle_component_id"]
            or details["source_pcm_sha256"] != details["oracle_slice_pcm_sha256"]
        ):
            raise ValueError("P1 source identity does not match oracle provenance")
    elif (
        any(value is not None for value in oracle_values) or details["oracle_event_ids"]
    ):
        raise ValueError("non-P1 decoder calls cannot claim oracle provenance")


def _validate_nemo_tdt_call_details(
    decision: str | None,
    details: Mapping[str, object],
) -> None:
    if set(details) != _NEMO_TDT_DECODER_CALL_KEYS:
        raise ValueError("NeMo TDT decoder call details have an invalid shape")
    try:
        call_mode = ArmKind(details["call_mode"])
    except (TypeError, ValueError) as exc:
        raise ValueError("details.call_mode is invalid") from exc
    if call_mode not in (ArmKind.P0, ArmKind.P1):
        raise ValueError("the NeMo TDT adapter runs only P0 and P1")
    if (
        details["decoder_contract_version"] != 1
        or details["mapping_version"] != 1
        or details["sample_rate"] != 16_000
    ):
        raise ValueError("NeMo TDT decoder contract is not version 1 at 16000 Hz")
    for key in (
        "decoder_input_pcm_sha256",
        "model_artifact_sha256",
        "source_pcm_sha256",
        "source_to_decoder_map_sha256",
    ):
        require_sha256(details[key], f"details.{key}")
    for key in ("source_artifact_id", "source_to_decoder_map_id"):
        require_uuid4(details[key], f"details.{key}")
    _validate_decoder_map(details)
    facts = details["nemo_tdt"]
    if not isinstance(facts, Mapping) or set(facts) != _NEMO_TDT_CALL_FACT_KEYS:
        raise ValueError("details.nemo_tdt has an invalid shape")
    if facts["model_type"] != _NEMO_TDT_MODEL_TYPE:
        raise ValueError("details.nemo_tdt.model_type is not nemo-conformer-tdt")
    if (
        _require_finite_number(
            facts["timestamp_frame_seconds"],
            "details.nemo_tdt.timestamp_frame_seconds",
        )
        <= 0.0
    ):
        raise ValueError("details.nemo_tdt.timestamp_frame_seconds must be positive")
    token_count = _require_plain_int(
        facts["token_count"],
        "details.nemo_tdt.token_count",
    )
    raw_count = _require_plain_int(
        details["raw_segment_count"],
        "details.raw_segment_count",
    )
    # One raw hypothesis per window: the decoded text of every token.
    if raw_count > 1 or (raw_count == 1 and token_count == 0):
        raise ValueError("a NeMo TDT window yields at most one tokenized hypothesis")
    if (raw_count == 0) != (decision == DiagnosticCode.DECODER_NO_YIELD.value):
        raise ValueError("decoder no-yield decision and raw count do not agree")
    if details["parent_call_record_id"] is not None:
        raise ValueError("NeMo TDT calls cannot declare a parent call")
    _validate_decoder_oracle_provenance(call_mode, details)


def _validate_raw_projection_details(details: Mapping[str, object]) -> None:
    """Check the projection fields every adapter's raw hypothesis carries."""

    require_uuid4(details["text_id"], "details.text_id")
    _require_plain_int(details["segment_ordinal"], "details.segment_ordinal")
    start = _require_plain_int(
        details["start_decoder_sample"],
        "details.start_decoder_sample",
    )
    end = _require_plain_int(
        details["end_decoder_sample"],
        "details.end_decoder_sample",
        minimum=1,
    )
    if end <= start:
        raise ValueError("raw hypothesis decoder span must be non-empty")
    raw_start = _require_finite_number(
        details["raw_start_seconds"],
        "details.raw_start_seconds",
    )
    raw_end = _require_finite_number(
        details["raw_end_seconds"],
        "details.raw_end_seconds",
    )
    if raw_start < 0.0 or raw_end <= raw_start:
        raise ValueError("raw hypothesis dependency seconds must form a span")
    _require_bool(details["timestamp_clamped"], "details.timestamp_clamped")
    synthetic = details["synthetic_pad_hypothesis"]
    _require_bool(synthetic, "details.synthetic_pad_hypothesis")
    anchor_timeline = details["synthetic_anchor_timeline_id"]
    anchor_sample = details["synthetic_anchor_sample"]
    if synthetic:
        require_uuid4(anchor_timeline, "details.synthetic_anchor_timeline_id")
        _require_plain_int(anchor_sample, "details.synthetic_anchor_sample")
    elif anchor_timeline is not None or anchor_sample is not None:
        raise ValueError("real raw hypotheses cannot carry a synthetic anchor")


def _validate_raw_hypothesis_details(details: Mapping[str, object]) -> None:
    if set(details) != _RAW_HYPOTHESIS_KEYS:
        raise ValueError("raw hypothesis details have an invalid shape")
    _validate_raw_projection_details(details)
    _require_probability(details["no_speech_prob"], "details.no_speech_prob")
    _require_finite_number(details["avg_logprob"], "details.avg_logprob")
    compression = _require_finite_number(
        details["compression_ratio"],
        "details.compression_ratio",
    )
    if compression < 0.0:
        raise ValueError("details.compression_ratio must be nonnegative")
    _require_finite_number(details["temperature"], "details.temperature")


def _validate_nemo_tdt_raw_details(details: Mapping[str, object]) -> None:
    if set(details) != _NEMO_TDT_RAW_HYPOTHESIS_KEYS:
        raise ValueError("NeMo TDT raw hypothesis details have an invalid shape")
    _validate_raw_projection_details(details)
    if details["segment_ordinal"] != 0:
        raise ValueError("a NeMo TDT window has one raw hypothesis at ordinal 0")
    facts = details["nemo_tdt"]
    if not isinstance(facts, Mapping) or set(facts) != _NEMO_TDT_RAW_FACT_KEYS:
        raise ValueError("details.nemo_tdt has an invalid shape")
    compression = _require_finite_number(
        facts["compression_ratio"],
        "details.nemo_tdt.compression_ratio",
    )
    if compression < 0.0:
        raise ValueError("details.nemo_tdt.compression_ratio must be nonnegative")
    last_token = _require_finite_number(
        facts["last_text_token_seconds"],
        "details.nemo_tdt.last_text_token_seconds",
    )
    if last_token < details["raw_start_seconds"]:
        raise ValueError("the last text token cannot precede the first")


def _require_exception_class(value: object) -> str:
    if type(value) is not str or not value.isidentifier():
        raise ValueError("exception_class must be a Python class name")
    return value


def _validate_canonical_payload(
    kind: RecordKind,
    decision: str | None,
    details: Mapping[str, object],
    observation_status: ObservationStatus,
    *,
    approved: Mapping[str, object] | None,
) -> None:
    """Enforce the closed, text-free metadata subset implemented by E0.

    ``approved`` is the run's declared contracts; records bound to a source or
    dependency contract are invalid without them.
    """

    allowed_decisions = _DECISION_VALUES_BY_KIND.get(kind, frozenset())
    if decision is not None and decision not in allowed_decisions:
        raise ValueError(f"{kind.value} decision must use its closed vocabulary")
    keys = set(details)
    if kind is RecordKind.ARM_STATUS:
        if keys not in (
            {"status"},
            {"status", "reason"},
            {"status", "reason", "exception_class"},
        ):
            raise ValueError("arm status details have an invalid shape")
        try:
            status = ArmStatus(details["status"])
        except (TypeError, ValueError) as exc:
            raise ValueError("arm status details contain an invalid status") from exc
        if status not in _TERMINAL_ARM_STATUSES or decision != status.value:
            raise ValueError("arm status decision and details do not agree")
        reason = details.get("reason")
        if status is ArmStatus.COMPLETE:
            if reason is not None:
                raise ValueError("complete arm status must not have a reason")
        elif reason not in _ARM_REASON_VALUES:
            raise ValueError("arm status reason must use the closed vocabulary")
        if "exception_class" in details:
            _require_exception_class(details["exception_class"])
        return
    if kind is RecordKind.DECODER_CALL:
        if details.get("decoder_adapter") == _NEMO_TDT:
            _validate_nemo_tdt_call_details(decision, details)
        else:
            _validate_decoder_call_details(decision, details, approved)
        return
    if kind is RecordKind.RAW_HYPOTHESIS:
        if keys == {"text_id"}:
            # Wire 1.0 records written before decoder instrumentation carried
            # only the private-text binding.  Keep them readable; new decoder
            # calls always emit the complete closed observation payload.
            require_uuid4(details["text_id"], "details.text_id")
        elif keys == _NEMO_TDT_RAW_HYPOTHESIS_KEYS:
            _validate_nemo_tdt_raw_details(details)
        else:
            _validate_raw_hypothesis_details(details)
        return
    if kind is RecordKind.WRAPPER_DECISION:
        if decision == TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value:
            if keys != {"text_id"}:
                raise ValueError("available wrapper hypotheses require text_id")
            require_uuid4(details["text_id"], "details.text_id")
        elif decision is None and keys == {"text_id"}:
            # Wire 1.0 records written before decoder instrumentation used the
            # private-text binding without a terminal decision.  Decoder-linked
            # records are still required to carry the closed decision vocabulary
            # by the cross-record semantic verifier below.
            require_uuid4(details["text_id"], "details.text_id")
        elif keys:
            raise ValueError("rejected or empty wrapper decisions cannot carry text_id")
        return
    if kind is RecordKind.FRAME:
        if keys != _FRAME_KEYS:
            raise ValueError("frame details have an invalid shape")
        _require_plain_int(details["frame_ordinal"], "details.frame_ordinal")
        probability = _require_probability(
            details["vad_probability"],
            "details.vad_probability",
        )
        silero = _require_probability(
            details["silero_probability"],
            "details.silero_probability",
        )
        term = _require_probability(details["sound_tag_term"], "details.sound_tag_term")
        if probability != max(silero, term):
            raise ValueError(
                "details.vad_probability is not the larger of "
                "silero_probability and sound_tag_term"
            )
        _require_bool(details["gate_in_speech_after"], "details.gate_in_speech_after")
        return
    if kind is RecordKind.GATE_TRANSITION:
        legacy_keys = {"in_speech_before", "in_speech_after"}
        extended_keys = legacy_keys | {"state_before", "state_after"}
        if keys not in (legacy_keys, extended_keys):
            raise ValueError("gate transition details have an invalid shape")
        before = _require_bool(
            details["in_speech_before"],
            "details.in_speech_before",
        )
        after = _require_bool(
            details["in_speech_after"],
            "details.in_speech_after",
        )
        if keys == extended_keys:
            before_state = _validate_gate_state(
                details["state_before"],
                "details.state_before",
            )
            after_state = _validate_gate_state(
                details["state_after"],
                "details.state_after",
            )
            if (
                before_state["in_speech"] is not before
                or after_state["in_speech"] is not after
            ):
                raise ValueError("gate transition booleans disagree with snapshots")
        if not before and after:
            if decision is not None:
                raise ValueError("gate-open transition must not have a decision")
        elif before and after:
            if decision != EndpointEmitReason.DIP_CUT.value:
                raise ValueError("open-to-open transition must be a dip cut")
        elif before and not after:
            if decision in (None, EndpointEmitReason.DIP_CUT.value):
                raise ValueError("gate-close transition needs a close reason")
        else:
            raise ValueError("gate transition must change or emit state")
        return
    if kind is RecordKind.WINDOW:
        legacy_keys = {"window_seq", "pcm_sha256", "forced", "dip_cut"}
        extended_keys = legacy_keys | {
            "state_before_emit",
            "state_after_emit",
        }
        if keys not in (legacy_keys, extended_keys):
            raise ValueError("window details have an invalid shape")
        _require_plain_int(details["window_seq"], "details.window_seq")
        require_sha256(details["pcm_sha256"], "details.pcm_sha256")
        forced = _require_bool(details["forced"], "details.forced")
        dip_cut = _require_bool(details["dip_cut"], "details.dip_cut")
        if forced and dip_cut:
            raise ValueError("window cannot be both forced and dip-cut")
        if forced:
            expected_decisions = {EndpointEmitReason.FORCE_SPLIT.value}
        elif dip_cut:
            expected_decisions = {EndpointEmitReason.DIP_CUT.value}
        elif keys == extended_keys:
            expected_decisions = {
                EndpointEmitReason.SILENCE_CLOSE.value,
                EndpointEmitReason.GLUE_TIMEOUT.value,
            }
        else:
            # Wire 1.0 endpoint records written before the typed production
            # reason existed could only classify a non-forced close as unknown.
            expected_decisions = {EndpointEmitReason.UNKNOWN.value}
        if decision not in expected_decisions:
            raise ValueError("window flags and decision do not agree")
        if keys == extended_keys:
            state_before = _validate_gate_state(
                details["state_before_emit"],
                "details.state_before_emit",
            )
            state_after = _validate_gate_state(
                details["state_after_emit"],
                "details.state_after_emit",
            )
            if state_before["in_speech"] is not True:
                raise ValueError("window must emit from an open gate")
            if state_after["in_speech"] is not dip_cut:
                raise ValueError("only a dip cut may leave the gate open")
            if not dip_cut and state_after["retained_frames"] != 0:
                raise ValueError("a true gate close must retain zero frames")
            if dip_cut and state_after["retained_frames"] < 1:
                raise ValueError("a dip cut must retain its live right piece")
        return
    if kind is RecordKind.MEASUREMENT:
        if decision is not None:
            raise ValueError("measurement records must not carry a decision")
        name = details.get("measurement_name")
        if type(name) is not str or name not in _MEASUREMENT_UNITS:
            raise ValueError("measurement name is not supported")
        # Device-wide VRAM is recorded as the arm's peak above its arm-start
        # reading, so the record names that scope and may be zero.
        shape = {"measurement_name", "unit"}
        if name == "peak_vram_bytes":
            shape.add("vram_scope")
        if keys not in (shape, shape | {"value"}):
            raise ValueError("measurement details have an invalid shape")
        if name == "peak_vram_bytes" and details["vram_scope"] != _VRAM_SCOPE:
            raise ValueError("peak_vram_bytes must use the device_delta scope")
        if details["unit"] != _MEASUREMENT_UNITS[name]:
            raise ValueError("measurement unit does not match its name")
        if observation_status is ObservationStatus.OBSERVED:
            if "value" not in details:
                raise ValueError("observed measurements require a value")
            value = _require_finite_number(
                details["value"],
                "details.value",
            )
            if (
                value < 0.0
                or (
                    name in {"peak_rss_bytes", "peak_vram_bytes"}
                    and type(details["value"]) is not int
                )
                or (name == "peak_rss_bytes" and value == 0)
            ):
                raise ValueError("observed measurement value is invalid")
        elif observation_status is ObservationStatus.INSTRUMENTATION_MISSING:
            if "value" in details:
                raise ValueError("missing instrumentation must not invent a value")
        else:
            raise ValueError("measurements must be observed or instrumentation_missing")
        return
    if kind is RecordKind.INTEGRITY and details:
        if decision is not None:
            raise ValueError("replay integrity record has no decision")
        if set(details) == _REPLAY_CONTRACT_KEYS:
            _validate_replay_contract_details(details, approved)
        elif set(details) == _REPLAY_TERMINAL_KEYS:
            _validate_replay_terminal_details(details)
        else:
            raise ValueError("integrity replay details have an invalid shape")
        return
    if details:
        raise ValueError(f"{kind.value} details schema is not implemented")


def _text_reference(
    record: EvidenceRecord | EvidenceRecordDraft,
) -> tuple[str, TextRole] | None:
    role = _TEXT_ROLE_BY_KIND.get(record.kind)
    text_id = record.details.get("text_id")
    if role is None or text_id is None:
        return None
    return require_uuid4(text_id, "details.text_id"), role


def _record_loss_codes(record: EvidenceRecord) -> tuple[LossCode, ...]:
    losses: list[LossCode] = []
    if (
        record.observation_status is ObservationStatus.RECORD_LOST
        or record.decision
        in {TerminalOutcome.RECORD_LOST.value, TerminalOutcome.CAPTURE_LOSS.value}
        or DiagnosticCode.CALLBACK_OVERFLOW in record.diagnostics
    ):
        losses.append(LossCode.RECORD_LOST)
    if DiagnosticCode.LEDGER_OVERFLOW in record.diagnostics:
        losses.append(LossCode.LEDGER_OVERFLOW)
    return tuple(losses)


class _AppendStream:
    def __init__(
        self,
        path: Path,
        name: StreamName,
        stream_id: str,
        classification: PrivateContentClass,
    ) -> None:
        self.path = path
        self.name = name
        self.stream_id = stream_id
        self.classification = classification
        self.count = 0
        self.first_seq: int | None = None
        self.last_seq: int | None = None
        self._sha256 = hashlib.sha256()
        self._stream: BinaryIO | None = path.open("xb")

    def append(
        self,
        payload: bytes,
        sequence: int,
        classification: PrivateContentClass,
    ) -> None:
        if self._stream is None:
            raise RuntimeError("stream is closed")
        written = self._stream.write(payload)
        if written != len(payload):
            raise OSError("short append-only stream write")
        self._stream.flush()
        self._sha256.update(payload)
        if self.first_seq is None:
            self.first_seq = sequence
        self.last_seq = sequence
        self.count += 1
        self.classification = inherited_classification(
            self.classification,
            (classification,),
        )

    def finish(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
        finally:
            self._stream.close()
            self._stream = None

    def close_without_fsync(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def intended_summary(self) -> StreamSummary:
        return StreamSummary(
            name=self.name,
            stream_id=self.stream_id,
            record_count=self.count,
            first_seq=self.first_seq,
            last_seq=self.last_seq,
            byte_sha256=self._sha256.hexdigest(),
            classification=self.classification,
        )


@dataclass(frozen=True, slots=True)
class _CanonicalScan:
    records: tuple[EvidenceRecord, ...]
    summary: StreamSummary | None
    arm_statuses: tuple[ArmTerminalStatus, ...]
    reasons: tuple[IntegrityFailureCode, ...]


@dataclass(frozen=True, slots=True)
class _PrivateScan:
    records: tuple[PrivateTextRecord, ...]
    summary: StreamSummary | None
    reasons: tuple[IntegrityFailureCode, ...]


def _complete_lines(
    raw: bytes,
) -> tuple[tuple[bytes, ...], tuple[IntegrityFailureCode, ...]]:
    if not raw:
        return (), ()
    reasons: list[IntegrityFailureCode] = []
    if not raw.endswith(b"\n"):
        reasons.append(IntegrityFailureCode.INVALID_OR_TRUNCATED_JSONL)
    lines = tuple(raw.splitlines(keepends=True))
    if any(
        line == b"\n" or not line.endswith(b"\n") or line.endswith(b"\r\n")
        for line in lines
    ):
        reasons.append(IntegrityFailureCode.INVALID_OR_TRUNCATED_JSONL)
    return lines, _unique_reasons(reasons)


def _read_stream(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _is_replay_contract(record: EvidenceRecord) -> bool:
    return (
        record.kind is RecordKind.INTEGRITY
        and set(record.details) == _REPLAY_CONTRACT_KEYS
    )


def _is_replay_terminal(record: EvidenceRecord) -> bool:
    return (
        record.kind is RecordKind.INTEGRITY
        and set(record.details) == _REPLAY_TERMINAL_KEYS
    )


def _compact_frame_record_spans(
    frames: Iterable[EvidenceRecord],
) -> tuple[SampleSpan, ...]:
    spans: list[SampleSpan] = []
    for frame in frames:
        if len(frame.source_spans) != 1:
            return ()
        spans.append(frame.source_spans[0])
    if not spans:
        return ()
    compacted: list[SampleSpan] = [spans[0]]
    for span in spans[1:]:
        previous = compacted[-1]
        if previous.timeline_id == span.timeline_id and previous.end == span.start:
            compacted[-1] = SampleSpan(
                previous.timeline_id,
                previous.start,
                span.end,
            )
        else:
            compacted.append(span)
    return tuple(compacted)


def _ordered_spans_within(
    spans: tuple[SampleSpan, ...],
    *,
    timeline_id: str,
    sample_count: int,
) -> bool:
    previous_end = -1
    for span in spans:
        if (
            span.timeline_id != timeline_id
            or span.end > sample_count
            or span.start < previous_end
        ):
            return False
        previous_end = span.end
    return True


def _replay_semantic_reasons(
    records: tuple[EvidenceRecord, ...],
) -> tuple[IntegrityFailureCode, ...]:
    reasons: list[IntegrityFailureCode] = []
    records_by_id = {record.record_id: record for record in records}
    contracts = tuple(record for record in records if _is_replay_contract(record))
    contract_ids = {record.record_id for record in contracts}
    all_terminals = tuple(record for record in records if _is_replay_terminal(record))

    for terminal in all_terminals:
        contract_id = terminal.details.get("replay_contract_id")
        if (
            type(contract_id) is not str
            or contract_id not in contract_ids
            or contract_id not in terminal.parent_ids
        ):
            reasons.append(IntegrityFailureCode.PARENT_LINK_INVALID)

    for contract in contracts:
        details = contract.details
        timeline_id = details.get("timeline_id")
        master_sample_count = details.get("master_sample_count")
        complete_frame_count = details.get("complete_frame_count")
        partial_frame_samples = details.get("partial_frame_samples")
        stop_value = details.get("stop_kind")
        if (
            type(timeline_id) is not str
            or type(master_sample_count) is not int
            or type(complete_frame_count) is not int
            or type(partial_frame_samples) is not int
            or master_sample_count < 1
            or complete_frame_count < 0
            or not 0 <= partial_frame_samples < 512
            or complete_frame_count * 512 + partial_frame_samples != master_sample_count
        ):
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
            continue
        try:
            require_uuid4(timeline_id, "details.timeline_id")
        except (TypeError, ValueError):
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
            continue
        try:
            stop_kind = ReplayStopKind(stop_value)
        except (TypeError, ValueError):
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
            continue
        if contract.stage is not Stage.INTEGRITY:
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
        if contract.classification is not PrivateContentClass.PRIVATE_AUDIO:
            reasons.append(IntegrityFailureCode.CLASSIFICATION_MISMATCH)

        related = tuple(
            record for record in records if contract.record_id in record.parent_ids
        )
        replay_related = tuple(
            record
            for record in related
            if record.kind
            in {
                RecordKind.FRAME,
                RecordKind.GATE_TRANSITION,
                RecordKind.WINDOW,
                RecordKind.DIAGNOSTIC,
                RecordKind.INTEGRITY,
            }
        )
        for record in replay_related:
            if record.arm_id != contract.arm_id:
                reasons.append(IntegrityFailureCode.RECORD_SCOPE_MISMATCH)
            if record.classification is not PrivateContentClass.PRIVATE_AUDIO:
                reasons.append(IntegrityFailureCode.CLASSIFICATION_MISMATCH)

        frames = tuple(
            record for record in replay_related if record.kind is RecordKind.FRAME
        )
        # A requested shutdown abandons the frames still in the sound-tag
        # assist's delay line unscored, so a cancelled replay records a prefix
        # of its frames; every other replay records them all.
        scored_frame_count = (
            min(len(frames), complete_frame_count)
            if stop_kind is ReplayStopKind.CANCELLED
            else complete_frame_count
        )
        if len(frames) != scored_frame_count:
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
        frames_by_ordinal: dict[int, EvidenceRecord] = {}
        for frame in frames:
            ordinal = frame.details.get("frame_ordinal")
            if (
                type(ordinal) is not int
                or not 0 <= ordinal < complete_frame_count
                or ordinal in frames_by_ordinal
            ):
                reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
                continue
            frames_by_ordinal[ordinal] = frame
            expected_span = SampleSpan(
                timeline_id,
                ordinal * 512,
                ordinal * 512 + 512,
            )
            if (
                frame.stage is not Stage.VAD
                or frame.parent_ids != (contract.record_id,)
                or frame.source_spans != (expected_span,)
            ):
                reasons.append(IntegrityFailureCode.PARENT_LINK_INVALID)
        if set(frames_by_ordinal) != set(range(scored_frame_count)):
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)

        windows = tuple(
            record for record in replay_related if record.kind is RecordKind.WINDOW
        )
        window_sequences = tuple(record.details.get("window_seq") for record in windows)
        if window_sequences != tuple(range(len(windows))):
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
        consumed_frame_ids: set[str] = set()
        for window in windows:
            frame_parents = tuple(
                records_by_id[parent_id]
                for parent_id in window.parent_ids[1:]
                if parent_id in records_by_id
                and records_by_id[parent_id].kind is RecordKind.FRAME
            )
            expected_parent_ids = (
                contract.record_id,
                *(frame.record_id for frame in frame_parents),
            )
            expected_spans = _compact_frame_record_spans(frame_parents)
            frame_ids = {frame.record_id for frame in frame_parents}
            frame_ordinals = tuple(
                frame.details.get("frame_ordinal") for frame in frame_parents
            )
            if (
                window.stage is not Stage.ENDPOINT
                or not frame_parents
                or window.parent_ids != expected_parent_ids
                or window.source_spans != expected_spans
                or not frame_ids.issubset({frame.record_id for frame in frames})
                or consumed_frame_ids.intersection(frame_ids)
                or any(type(item) is not int for item in frame_ordinals)
                or frame_ordinals != tuple(sorted(frame_ordinals))
            ):
                reasons.append(IntegrityFailureCode.PARENT_LINK_INVALID)
            consumed_frame_ids.update(frame_ids)

        gates = tuple(
            record
            for record in replay_related
            if record.kind is RecordKind.GATE_TRANSITION
        )
        frame_ordinals_by_id = {
            frame.record_id: ordinal for ordinal, frame in frames_by_ordinal.items()
        }
        transitions_by_ordinal: dict[int, EvidenceRecord] = {}
        for gate in gates:
            ordinal = (
                frame_ordinals_by_id.get(gate.parent_ids[1])
                if len(gate.parent_ids) > 1
                else None
            )
            if (
                gate.stage is not Stage.ENDPOINT
                or not gate.parent_ids
                or gate.parent_ids[0] != contract.record_id
                or ordinal is None
                or ordinal in transitions_by_ordinal
                or not _ordered_spans_within(
                    gate.source_spans,
                    timeline_id=timeline_id,
                    sample_count=master_sample_count,
                )
            ):
                reasons.append(IntegrityFailureCode.PARENT_LINK_INVALID)
            if ordinal is not None:
                transitions_by_ordinal.setdefault(ordinal, gate)
        # The gate opens closed; each frame's recorded state after its push
        # holds the last transition's ``in_speech_after`` until the next one.
        in_speech: object = False
        for ordinal in sorted(frames_by_ordinal):
            transition = transitions_by_ordinal.get(ordinal)
            if transition is not None:
                in_speech = transition.details.get("in_speech_after")
            recorded = frames_by_ordinal[ordinal].details.get("gate_in_speech_after")
            if type(recorded) is not bool or recorded is not in_speech:
                reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
                break

        diagnostics = tuple(
            record for record in replay_related if record.kind is RecordKind.DIAGNOSTIC
        )
        actual_codes: list[str] = []
        for diagnostic in diagnostics:
            try:
                code = DiagnosticCode(diagnostic.decision)
            except (TypeError, ValueError):
                reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
                continue
            actual_codes.append(code.value)
            if (
                diagnostic.stage is not Stage.ENDPOINT
                or diagnostic.parent_ids != (contract.record_id,)
                or diagnostic.diagnostics != (code,)
                or not _ordered_spans_within(
                    diagnostic.source_spans,
                    timeline_id=timeline_id,
                    sample_count=master_sample_count,
                )
            ):
                reasons.append(IntegrityFailureCode.PARENT_LINK_INVALID)

        terminals = tuple(
            record for record in replay_related if _is_replay_terminal(record)
        )
        if len(terminals) != 1:
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
            continue
        terminal = terminals[0]
        terminal_details = terminal.details
        pending_span_count = terminal_details.get("pending_span_count")
        has_partial_frame = terminal_details.get("has_partial_frame")
        diagnostic_codes = terminal_details.get("diagnostic_codes")
        if (
            type(pending_span_count) is not int
            or pending_span_count < 0
            or type(has_partial_frame) is not bool
            or not isinstance(diagnostic_codes, tuple)
        ):
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
            continue
        declared_codes = diagnostic_codes
        expected_terminal_parents = (
            contract.record_id,
            *(record.record_id for record in windows),
            *(record.record_id for record in gates),
            *(record.record_id for record in diagnostics),
        )
        if (
            terminal.stage is not Stage.INTEGRITY
            or terminal_details.get("replay_contract_id") != contract.record_id
            or terminal_details.get("stop_kind") != stop_kind.value
            or terminal_details.get("window_count") != len(windows)
            or terminal_details.get("gate_transition_count") != len(gates)
            or terminal.parent_ids != expected_terminal_parents
            or declared_codes != tuple(actual_codes)
            or has_partial_frame != (partial_frame_samples > 0)
            or len(terminal.source_spans) != pending_span_count + int(has_partial_frame)
        ):
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)

        pending_spans = terminal.source_spans[:pending_span_count]
        partial_spans = terminal.source_spans[pending_span_count:]
        complete_samples = complete_frame_count * 512
        if not _ordered_spans_within(
            pending_spans,
            timeline_id=timeline_id,
            sample_count=complete_samples,
        ) or any(span.start % 512 or span.end % 512 for span in pending_spans):
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
        # The unscored frames of a cancelled replay are pending shutdown.
        if scored_frame_count < complete_frame_count and not (
            pending_spans
            and pending_spans[-1].start <= scored_frame_count * 512
            and pending_spans[-1].end == complete_samples
        ):
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
        window_spans = tuple(span for window in windows for span in window.source_spans)
        if any(
            pending.timeline_id == emitted.timeline_id
            and pending.start < emitted.end
            and emitted.start < pending.end
            for pending in pending_spans
            for emitted in window_spans
        ):
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
        expected_partial = (
            (
                SampleSpan(
                    timeline_id,
                    complete_samples,
                    master_sample_count,
                ),
            )
            if partial_frame_samples
            else ()
        )
        if partial_spans != expected_partial:
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)

        affected = bool(pending_spans or partial_spans)
        expected_codes: list[str] = []
        if stop_kind is ReplayStopKind.CANCELLED:
            if affected:
                expected_codes.append(DiagnosticCode.ENDPOINT_PENDING_SHUTDOWN.value)
            expected_codes.append(DiagnosticCode.SHUTDOWN_CANCELLED.value)
        else:
            if pending_spans:
                expected_codes.append(DiagnosticCode.ENDPOINT_PENDING_EOF.value)
            if partial_spans:
                expected_codes.append(DiagnosticCode.ENDPOINT_PARTIAL_FRAME_EOF.value)
            if not windows and not affected:
                expected_codes.append(DiagnosticCode.VAD_NOT_DETECTED.value)
        if actual_codes != expected_codes:
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)

    return _unique_reasons(reasons)


def _decoder_payload_valid(
    record: EvidenceRecord,
    approved: Mapping[str, object] | None,
) -> bool:
    try:
        _validate_canonical_payload(
            record.kind,
            record.decision,
            record.details,
            record.observation_status,
            approved=approved,
        )
    except (TypeError, ValueError):
        return False
    return True


def _is_nemo_tdt_call(call: EvidenceRecord) -> bool:
    return call.details.get("decoder_adapter") == _NEMO_TDT


def decoder_sample_bounds(
    start_seconds: float,
    end_seconds: float,
    sample_rate: int,
    decoder_samples: int,
) -> tuple[int, int, bool]:
    """Round dependency seconds to decoder samples, clamped to the input.

    faster-whisper 1.2.1 builds segment times from timestamp tokens and never
    limits them to the audio, so a hallucinated segment can end, or even start,
    after the decoder input. The end is clamped to ``decoder_samples``; a start
    at or past the end becomes the final input sample so the span stays
    non-empty. The flag reports whether either boundary moved.
    """

    start = math.floor(start_seconds * sample_rate + 0.5)
    end = math.floor(end_seconds * sample_rate + 0.5)
    if start < 0 or end <= start:
        raise ValueError("decoder timestamps must round to a nonnegative non-empty span")
    if decoder_samples < 1:
        raise ValueError("decoder input must contain samples")
    clamped_start = min(start, decoder_samples - 1)
    clamped_end = min(end, decoder_samples)
    return clamped_start, clamped_end, (clamped_start, clamped_end) != (start, end)


def stochastic_tail_start(temperatures: Iterable[float]) -> int:
    """Return the ordinal of a call's first temperature-sampled raw segment.

    A segment decoded at temperature > 0 came from faster-whisper's sampling
    fallback. CTranslate2 seeds its sampling generator once per thread, so
    reseeding before a call cannot repeat the draw, and later windows seek from
    the sampled output. Earlier segments came from greedy decoding and must
    reproduce exactly; this segment and every later one are stochastic.
    """

    values = tuple(temperatures)
    for ordinal, temperature in enumerate(values):
        if temperature > 0.0:
            return ordinal
    return len(values)


def decoder_call_diagnostics(
    temperatures: Sequence[float],
) -> tuple[DiagnosticCode, ...]:
    """Diagnostics carried by a decoder-call record, given its raw yields."""

    if not temperatures:
        return (DiagnosticCode.UNKNOWN_DECODER_GATE,)
    if stochastic_tail_start(temperatures) < len(temperatures):
        return (DiagnosticCode.DECODER_STOCHASTIC_FALLBACK,)
    return ()


def raw_hypothesis_diagnostics(
    *,
    timestamp_clamped: bool,
    stochastic: bool,
) -> tuple[DiagnosticCode, ...]:
    """Diagnostics carried by one raw-hypothesis record."""

    return tuple(
        code
        for code, present in (
            (DiagnosticCode.DECODER_TIMESTAMP_CLAMPED, timestamp_clamped),
            (DiagnosticCode.DECODER_STOCHASTIC_FALLBACK, stochastic),
        )
        if present
    )


@dataclass(frozen=True, slots=True)
class TextOnlyGate:
    """Production's text-derived wrapper gates applied to one decoder window."""

    assembled_text: str
    compression_ratio: float
    compression_rejected: bool


def text_only_wrapper_gate(raw_text: str) -> TextOnlyGate:
    """Apply the wrapper gates a decoder without Whisper confidence can meet.

    ``FasterWhisperASR.transcribe`` first drops a segment on Whisper's
    confidence (``no_speech_prob`` alone above ``no_speech_only_threshold``
    when that is set, then the paired ``no_speech_prob``/``avg_logprob``
    gate), then rejects assembled text of at least
    ``_MIN_COMPRESSION_GATE_LEN`` characters whose window compression
    ratio exceeds ``_COMPRESSION_RATIO_THRESHOLD``; faster-whisper computes
    that ratio with ``get_compression_ratio`` over the window's stripped text.
    A non-Whisper decoder reports no confidence pair, so only the compression
    gate and the empty-text outcome apply, with the same function and limits.
    The candidate adapter and the verifier both call this.
    """

    if type(raw_text) is not str:
        raise TypeError("raw_text must be str")
    # faster-whisper must load behind production's NumPy-only audio shim.
    import_whisper_model()
    from faster_whisper.transcribe import get_compression_ratio

    assembled = raw_text.strip()
    ratio = float(get_compression_ratio(assembled))
    return TextOnlyGate(
        assembled_text=assembled,
        compression_ratio=ratio,
        compression_rejected=(
            len(assembled) >= _MIN_COMPRESSION_GATE_LEN
            and ratio > _COMPRESSION_RATIO_THRESHOLD
        ),
    )



def _decoder_map_intervals(
    call: EvidenceRecord,
) -> tuple[tuple[int, int, SampleSpan | None], ...]:
    result: list[tuple[int, int, SampleSpan | None]] = []
    for interval in call.details["source_to_decoder_map"]:
        source_wire = interval["source_span"]
        source = (
            None
            if source_wire is None
            else SampleSpan(
                source_wire["timeline_id"],
                source_wire["start"],
                source_wire["end"],
            )
        )
        result.append((interval["decoder_start"], interval["decoder_end"], source))
    return tuple(result)


def _project_raw_semantics(
    call: EvidenceRecord,
    raw: EvidenceRecord,
) -> tuple[tuple[SampleSpan, ...], str | None, int | None]:
    start = raw.details["start_decoder_sample"]
    end = raw.details["end_decoder_sample"]
    intervals = _decoder_map_intervals(call)
    projected: list[SampleSpan] = []
    for decoder_start, decoder_end, source in intervals:
        if source is None:
            continue
        overlap_start = max(start, decoder_start)
        overlap_end = min(end, decoder_end)
        if overlap_end <= overlap_start:
            continue
        candidate = SampleSpan(
            source.timeline_id,
            source.start + overlap_start - decoder_start,
            source.start + overlap_end - decoder_start,
        )
        if (
            projected
            and projected[-1].timeline_id == candidate.timeline_id
            and projected[-1].end == candidate.start
        ):
            previous = projected[-1]
            projected[-1] = SampleSpan(
                previous.timeline_id,
                previous.start,
                candidate.end,
            )
        else:
            projected.append(candidate)
    if projected:
        return tuple(projected), None, None
    midpoint = (start + end) / 2.0
    candidates: list[tuple[float, int, str, int]] = []
    for decoder_start, decoder_end, source in intervals:
        if source is None:
            continue
        candidates.extend(
            (
                (
                    abs(midpoint - decoder_start),
                    decoder_start,
                    source.timeline_id,
                    source.start,
                ),
                (
                    abs(midpoint - decoder_end),
                    decoder_end,
                    source.timeline_id,
                    source.end,
                ),
            )
        )
    _, _, timeline_id, sample = min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return (), timeline_id, sample


def _semantic_span_union(spans: Iterable[SampleSpan]) -> tuple[SampleSpan, ...]:
    ordered = sorted(spans, key=lambda span: (span.timeline_id, span.start, span.end))
    result: list[SampleSpan] = []
    for span in ordered:
        if (
            result
            and result[-1].timeline_id == span.timeline_id
            and span.start <= result[-1].end
        ):
            previous = result[-1]
            result[-1] = SampleSpan(
                previous.timeline_id,
                previous.start,
                max(previous.end, span.end),
            )
        else:
            result.append(span)
    return tuple(result)


def _wrapper_positions_by_parent(
    records: Sequence[EvidenceRecord],
) -> dict[str, list[int]]:
    """Index every wrapper decision's ledger position under each of its parents."""

    wrappers_by_parent: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        if record.kind is RecordKind.WRAPPER_DECISION:
            for parent_id in set(record.parent_ids):
                wrappers_by_parent.setdefault(parent_id, []).append(position)
    return wrappers_by_parent


def _wrapper_decisions_under(
    records: Sequence[EvidenceRecord],
    wrappers_by_parent: Mapping[str, Sequence[int]],
    parent_ids: Iterable[str],
) -> tuple[EvidenceRecord, ...]:
    """Wrapper decisions with any parent in ``parent_ids``, in ledger order."""

    positions = {
        position
        for parent_id in parent_ids
        for position in wrappers_by_parent.get(parent_id, ())
    }
    return tuple(records[position] for position in sorted(positions))


def _no_speech_only_threshold(start: RunStartManifest) -> float | None:
    """The run's ``asr.faster_whisper.no_speech_only_threshold``.

    Read from the ``effective_config`` that ``run-start.json`` records, the
    config the run built its wrapper from. A run recorded before the setting
    existed, or with it null, applies only the paired confidence gate. Raises
    ``ValueError`` when the value is not a probability in (0, 1]. 1.0, like
    null, applies only the paired confidence gate.
    """

    config = start.details.get("effective_config")
    asr = config.get("asr") if isinstance(config, Mapping) else None
    whisper = asr.get("faster_whisper") if isinstance(asr, Mapping) else None
    value = (
        whisper.get("no_speech_only_threshold")
        if isinstance(whisper, Mapping)
        else None
    )
    if value is None:
        return None
    if type(value) not in (int, float) or not 0.0 < value <= 1.0:
        raise ValueError("effective no_speech_only_threshold must be in (0, 1]")
    return float(value)


def _whisper_confidence_rejected(
    no_speech_prob: float,
    avg_logprob: float,
    no_speech_only_threshold: float | None,
) -> bool:
    """The wrapper's confidence gate: ``no_speech_prob`` alone above the
    run's no-speech-only threshold, else the paired 0.6 / -1.0 gate."""

    if (
        no_speech_only_threshold is not None
        and no_speech_prob > no_speech_only_threshold
    ):
        return True
    return no_speech_prob > 0.6 and avg_logprob < -1.0


def _decoder_semantic_reasons(
    records: tuple[EvidenceRecord, ...],
    start: RunStartManifest,
) -> tuple[IntegrityFailureCode, ...]:
    """Verify cross-record decoder invariants that shape checks cannot prove."""

    invalid = IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID
    reasons: list[IntegrityFailureCode] = []
    approved = _approved_contracts(start)
    try:
        no_speech_only_threshold = _no_speech_only_threshold(start)
    except ValueError:
        reasons.append(invalid)
        no_speech_only_threshold = None
    definitions = {
        definition.arm_id: definition for definition in start.arm_definitions
    }
    calls = {
        record.record_id: record
        for record in records
        if record.kind is RecordKind.DECODER_CALL
        and _decoder_payload_valid(record, approved)
    }
    decoder_arm_ids = {
        record.arm_id for record in calls.values() if record.arm_id is not None
    }
    p1_components_by_arm: dict[str, dict[str, Mapping[str, object]]] = {}
    for definition in start.arm_definitions:
        if definition.kind is not ArmKind.P1:
            continue
        decoder_contract_declared = definition.arm_id in decoder_arm_ids or any(
            key in definition.parameters
            for key in (
                "decoder_random_seed",
                "model_artifact_sha256",
                "oracle_components",
                "oracle_component_id",
                "oracle_component_map_sha256",
                "oracle_event_ids",
                "oracle_reference_set_id",
                "oracle_reference_set_sha256",
            )
        )
        if not decoder_contract_declared:
            # Wire 1.0 allowed generic arm ledgers before decoder provenance was
            # implemented.  They remain readable; any decoder-bearing or newly
            # declared P1 arm is held to the complete oracle partition below.
            continue
        raw_components = definition.parameters.get("oracle_components")
        if raw_components is None:
            legacy_keys = {
                "oracle_component_id",
                "oracle_component_map_sha256",
                "oracle_event_ids",
                "oracle_reference_set_id",
                "oracle_reference_set_sha256",
            }
            if not legacy_keys.issubset(definition.parameters):
                reasons.append(invalid)
                p1_components_by_arm[definition.arm_id] = {}
                continue
            components: tuple[object, ...] = (
                {key: definition.parameters[key] for key in legacy_keys},
            )
        elif isinstance(raw_components, tuple):
            components = raw_components
        else:
            reasons.append(invalid)
            p1_components_by_arm[definition.arm_id] = {}
            continue
        declared: dict[str, Mapping[str, object]] = {}
        declared_event_ids: set[str] = set()
        reference_identity: tuple[str, str] | None = None
        expected_keys = {
            "oracle_component_id",
            "oracle_component_map_sha256",
            "oracle_event_ids",
            "oracle_reference_set_id",
            "oracle_reference_set_sha256",
        }
        for component in components:
            if not isinstance(component, Mapping) or set(component) != expected_keys:
                reasons.append(invalid)
                continue
            component_id = component.get("oracle_component_id")
            event_ids = component.get("oracle_event_ids")
            reference_id = component.get("oracle_reference_set_id")
            reference_sha256 = component.get("oracle_reference_set_sha256")
            try:
                component_id = require_uuid4(
                    component_id,
                    "oracle_component_id",
                )
                require_sha256(
                    component.get("oracle_component_map_sha256"),
                    "oracle_component_map_sha256",
                )
                if not isinstance(event_ids, tuple) or not event_ids:
                    raise ValueError("oracle_event_ids must be a nonempty tuple")
                event_ids = tuple(
                    require_uuid4(event_id, "oracle_event_id")
                    for event_id in event_ids
                )
                reference_id = require_uuid4(
                    reference_id,
                    "oracle_reference_set_id",
                )
                reference_sha256 = require_sha256(
                    reference_sha256,
                    "oracle_reference_set_sha256",
                )
            except (TypeError, ValueError):
                reasons.append(invalid)
                continue
            if (
                component_id in declared
                or len(event_ids) != len(set(event_ids))
                or declared_event_ids.intersection(event_ids)
            ):
                reasons.append(invalid)
                continue
            component_reference = (reference_id, reference_sha256)
            if reference_identity is None:
                reference_identity = component_reference
            elif reference_identity != component_reference:
                reasons.append(invalid)
                continue
            declared[component_id] = component
            declared_event_ids.update(event_ids)
        if not declared:
            reasons.append(invalid)
        p1_components_by_arm[definition.arm_id] = declared
    seen_p1_components: dict[str, set[str]] = {
        arm_id: set() for arm_id in p1_components_by_arm
    }
    raw_by_call: dict[str, list[EvidenceRecord]] = {
        record_id: [] for record_id in calls
    }
    for record in records:
        if (
            record.kind is not RecordKind.RAW_HYPOTHESIS
            or set(record.details)
            not in (_RAW_HYPOTHESIS_KEYS, _NEMO_TDT_RAW_HYPOTHESIS_KEYS)
            or not _decoder_payload_valid(record, approved)
        ):
            continue
        if len(record.parent_ids) != 1 or record.parent_ids[0] not in calls:
            reasons.append(invalid)
            continue
        call = calls[record.parent_ids[0]]
        if _is_nemo_tdt_call(call) != (
            set(record.details) == _NEMO_TDT_RAW_HYPOTHESIS_KEYS
        ):
            # A raw hypothesis must carry its own decoder's observation shape.
            reasons.append(invalid)
            continue
        if record.arm_id != call.arm_id:
            reasons.append(invalid)
        raw_by_call[call.record_id].append(record)

    input_identity_keys = (
        "decoder_contract_version",
        "decoder_input_pcm_sha256",
        "decoder_input_samples",
        "decoder_random_seed",
        "delegate_class_sha256",
        "delegate_instance_id",
        "mapping_version",
        "model_artifact_sha256",
        "sample_rate",
        "source_artifact_id",
        "source_pcm_sha256",
        "source_to_decoder_map",
        "source_to_decoder_map_id",
        "source_to_decoder_map_sha256",
        "synthetic_left_pad_samples",
        "synthetic_right_pad_samples",
    )
    raw_parity_keys = (
        "avg_logprob",
        "compression_ratio",
        "end_decoder_sample",
        "no_speech_prob",
        "raw_end_seconds",
        "raw_start_seconds",
        "segment_ordinal",
        "start_decoder_sample",
        "synthetic_anchor_sample",
        "synthetic_anchor_timeline_id",
        "synthetic_pad_hypothesis",
        "temperature",
        "timestamp_clamped",
    )
    terminal_decisions = {
        DiagnosticCode.WRAPPER_COMPRESSION_REJECT.value,
        DiagnosticCode.WRAPPER_EMPTY.value,
        TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value,
    }
    wrappers_by_parent = _wrapper_positions_by_parent(records)

    for call_id, call in calls.items():
        call_raw = sorted(
            raw_by_call[call_id],
            key=lambda record: record.details["segment_ordinal"],
        )
        ordinals = tuple(record.details["segment_ordinal"] for record in call_raw)
        if len(call_raw) != call.details["raw_segment_count"] or ordinals != tuple(
            range(len(call_raw))
        ):
            reasons.append(invalid)
        nemo_tdt = _is_nemo_tdt_call(call)
        if nemo_tdt:
            # Greedy TDT decoding has no hidden gate and no sampling fallback.
            tail = len(call_raw)
            expected_call_diagnostics: tuple[DiagnosticCode, ...] = ()
        else:
            temperatures = tuple(record.details["temperature"] for record in call_raw)
            tail = stochastic_tail_start(temperatures)
            expected_call_diagnostics = decoder_call_diagnostics(temperatures)
        if call.diagnostics != expected_call_diagnostics:
            reasons.append(invalid)

        mode = ArmKind(call.details["call_mode"])
        definition = definitions.get(call.arm_id)
        if nemo_tdt:
            # CTranslate2's sampling seed has no meaning for greedy TDT decoding.
            adapter_declared = (
                definition is not None
                and definition.parameters.get("decoder_adapter") == _NEMO_TDT
                and "decoder_random_seed" not in definition.parameters
            )
        else:
            adapter_declared = (
                definition is not None
                and approved is not None
                and "decoder_adapter" not in definition.parameters
                and definition.parameters.get("decoder_random_seed")
                == approved["decoder_random_seed"]
            )
        if (
            definition is None
            or definition.kind is not mode
            or not adapter_declared
            or definition.parameters.get("model_artifact_sha256")
            != call.details["model_artifact_sha256"]
        ):
            reasons.append(invalid)
        if definition is not None and mode is ArmKind.P0:
            if definition.parent_arm_id is not None or definition.declared_changes:
                reasons.append(invalid)
        if definition is not None and mode is ArmKind.P1:
            parent_definition = definitions.get(definition.parent_arm_id)
            if (
                parent_definition is None
                or parent_definition.kind is not ArmKind.P0
                or definition.declared_changes != ("boundary_mode",)
            ):
                reasons.append(invalid)
        if mode is ArmKind.P1 and definition is not None:
            component_id = call.details["oracle_component_id"]
            declared = p1_components_by_arm.get(definition.arm_id, {})
            expected = declared.get(component_id)
            actual = {
                "oracle_component_id": component_id,
                "oracle_component_map_sha256": call.details[
                    "oracle_component_map_sha256"
                ],
                "oracle_event_ids": call.details["oracle_event_ids"],
                "oracle_reference_set_id": call.details["oracle_reference_set_id"],
                "oracle_reference_set_sha256": call.details[
                    "oracle_reference_set_sha256"
                ],
            }
            seen = seen_p1_components.setdefault(definition.arm_id, set())
            if expected != actual or component_id in seen:
                reasons.append(invalid)
            else:
                seen.add(component_id)

        expected_call_spans = tuple(
            source
            for _, _, source in _decoder_map_intervals(call)
            if source is not None
        )
        if call.source_spans != expected_call_spans:
            reasons.append(invalid)
        for ordinal, raw in enumerate(call_raw):
            try:
                expected_spans, anchor_timeline, anchor_sample = _project_raw_semantics(
                    call, raw
                )
                bounds = decoder_sample_bounds(
                    raw.details["raw_start_seconds"],
                    raw.details["raw_end_seconds"],
                    call.details["sample_rate"],
                    call.details["decoder_input_samples"],
                )
                if (
                    bounds
                    != (
                        raw.details["start_decoder_sample"],
                        raw.details["end_decoder_sample"],
                        raw.details["timestamp_clamped"],
                    )
                    or raw.diagnostics
                    != raw_hypothesis_diagnostics(
                        timestamp_clamped=raw.details["timestamp_clamped"],
                        stochastic=ordinal >= tail,
                    )
                    or raw.source_spans != expected_spans
                    or (
                        nemo_tdt
                        and raw.details["raw_end_seconds"]
                        != raw.details["nemo_tdt"]["last_text_token_seconds"]
                        + call.details["nemo_tdt"]["timestamp_frame_seconds"]
                    )
                    or raw.details["synthetic_pad_hypothesis"] != (not expected_spans)
                    or raw.details["synthetic_anchor_timeline_id"] != anchor_timeline
                    or raw.details["synthetic_anchor_sample"] != anchor_sample
                ):
                    raise ValueError("raw projection mismatch")
            except (TypeError, ValueError):
                reasons.append(invalid)

        parent_call_id = call.details["parent_call_record_id"]
        if mode in (ArmKind.P2A, ArmKind.P2B):
            parent = calls.get(parent_call_id)
            expected_parent_mode = ArmKind.P0 if mode is ArmKind.P2A else ArmKind.P2A
            expected_declared_changes = (
                ("decoder.gates.no_speech_threshold",) if mode is ArmKind.P2B else ()
            )
            if (
                parent is None
                or call.parent_ids != (parent_call_id,)
                or parent.arm_id == call.arm_id
                or parent.details.get("call_mode") != expected_parent_mode.value
                or definition is None
                or definition.parent_arm_id != parent.arm_id
                or definition.declared_changes != expected_declared_changes
                or any(
                    call.details[key] != parent.details[key]
                    for key in input_identity_keys
                )
            ):
                reasons.append(invalid)
                continue
            if mode is ArmKind.P2A:
                parent_raw = sorted(
                    raw_by_call[parent.record_id],
                    key=lambda record: record.details["segment_ordinal"],
                )
                parent_tail = stochastic_tail_start(
                    record.details["temperature"] for record in parent_raw
                )
                # Greedy segments must reproduce exactly; both calls must reach
                # their stochastic fallback, if any, at the same ordinal.
                parity_matches = (
                    call.details["effective_arguments_sha256"]
                    == parent.details["effective_arguments_sha256"]
                    and tail == parent_tail
                    and (tail < len(call_raw)) == (parent_tail < len(parent_raw))
                    and all(
                        actual.source_spans == expected.source_spans
                        and all(
                            actual.details[key] == expected.details[key]
                            for key in raw_parity_keys
                        )
                        for expected, actual in zip(
                            parent_raw[:parent_tail],
                            call_raw[:tail],
                        )
                    )
                )
                if not parity_matches:
                    reasons.append(invalid)
            else:
                actual_arguments = call.details["effective_arguments"]
                parent_arguments = parent.details["effective_arguments"]
                changed = tuple(
                    key
                    for key in actual_arguments
                    if actual_arguments[key] != parent_arguments[key]
                )
                if changed != ("no_speech_threshold",):
                    reasons.append(invalid)
        elif call.parent_ids or parent_call_id is not None:
            reasons.append(invalid)

        # Only Whisper reports the confidence pair; a NeMo TDT window has none.
        rejected_raw_ids = tuple(
            raw.record_id
            for raw in call_raw
            if not nemo_tdt
            and _whisper_confidence_rejected(
                raw.details["no_speech_prob"],
                raw.details["avg_logprob"],
                no_speech_only_threshold,
            )
        )
        confidence_records = tuple(
            record
            for record in _wrapper_decisions_under(
                records,
                wrappers_by_parent,
                {raw.record_id for raw in call_raw},
            )
            if record.decision == DiagnosticCode.WRAPPER_CONFIDENCE_GATE.value
            and len(record.parent_ids) == 1
        )
        if tuple(
            record.parent_ids[0] for record in confidence_records
        ) != rejected_raw_ids or any(
            record.arm_id != call.arm_id for record in confidence_records
        ):
            reasons.append(invalid)
        kept_raw = tuple(
            raw for raw in call_raw if raw.record_id not in set(rejected_raw_ids)
        )
        related_ids = {
            call_id,
            *(record.record_id for record in call_raw),
            *(record.record_id for record in confidence_records),
        }
        terminal_records = tuple(
            candidate
            for candidate in _wrapper_decisions_under(
                records,
                wrappers_by_parent,
                related_ids,
            )
            if candidate.decision in terminal_decisions
            and all(parent_id in related_ids for parent_id in candidate.parent_ids)
        )
        if len(terminal_records) != 1:
            reasons.append(invalid)
            continue
        terminal = terminal_records[0]
        if terminal.arm_id != call.arm_id:
            reasons.append(invalid)
        if terminal.decision == DiagnosticCode.WRAPPER_EMPTY.value:
            if kept_raw:
                expected_parents = tuple(record.record_id for record in kept_raw)
            elif confidence_records:
                expected_parents = tuple(
                    record.record_id for record in confidence_records
                )
            else:
                expected_parents = (call_id,)
        else:
            expected_parents = tuple(record.record_id for record in kept_raw)
        expected_sources = _semantic_span_union(
            span for record in kept_raw for span in record.source_spans
        )
        if (
            terminal.parent_ids != expected_parents
            or terminal.source_spans != expected_sources
        ):
            reasons.append(invalid)

    complete_arm_ids = {
        record.arm_id
        for record in records
        if record.kind is RecordKind.ARM_STATUS
        and record.decision == ArmStatus.COMPLETE.value
        and record.arm_id is not None
    }
    for arm_id, declared in p1_components_by_arm.items():
        seen = seen_p1_components.get(arm_id, set())
        if not seen.issubset(declared) or (
            arm_id in complete_arm_ids and seen != set(declared)
        ):
            reasons.append(invalid)

    return _unique_reasons(reasons)


def _capture_replay_semantic_reasons(
    records: tuple[EvidenceRecord, ...],
    start: RunStartManifest,
) -> tuple[IntegrityFailureCode, ...]:
    """Bind every replay contract to the run's production-capture transform."""

    contracts = tuple(record for record in records if _is_replay_contract(record))
    if not contracts:
        return ()
    invalid = (IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID,)
    provenance = start.details.get("capture_replay")
    try:
        provenance_sha256 = capture_replay_provenance_sha256(provenance)
    except (TypeError, ValueError):
        return invalid
    if provenance["canonical_pcm_sha256"] != start.details.get("master_pcm_sha256"):
        return invalid
    for contract in contracts:
        details = contract.details
        if (
            details.get("capture_replay_sha256") != provenance_sha256
            or details.get("replay_pcm_sha256") != provenance["output_pcm_sha256"]
            or details.get("master_pcm_sha256") != provenance["canonical_pcm_sha256"]
            or details.get("timeline_id") != provenance["timeline_id"]
            or details.get("master_sample_count") != provenance["output_sample_count"]
        ):
            return invalid
    return ()


def _vad_artifact_semantic_reasons(
    records: tuple[EvidenceRecord, ...],
    start: RunStartManifest,
) -> tuple[IntegrityFailureCode, ...]:
    """Bind replay contracts and arm profiles to the run's declared VAD bytes.

    A run start without ``vad_artifact`` predates candidate selection, so its
    replay may only have used the shipped Silero pin.
    """

    invalid = (IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID,)
    declared = start.details.get("vad_artifact")
    declared_by_arms = tuple(
        definition.parameters.get("vad_artifact_sha256")
        for definition in start.arm_definitions
    )
    if declared is None:
        if any(value is not None for value in declared_by_arms):
            return invalid
        expected = SILERO_SHA256
    else:
        if not isinstance(declared, Mapping) or set(declared) != _VAD_ARTIFACT_KEYS:
            return invalid
        try:
            expected = require_sha256(declared["sha256"], "vad_artifact.sha256")
            _require_plain_int(
                declared["byte_count"], "vad_artifact.byte_count", minimum=1
            )
            production_pin = _require_bool(
                declared["production_pin"], "vad_artifact.production_pin"
            )
        except (TypeError, ValueError):
            return invalid
        if production_pin is not (expected == SILERO_SHA256) or any(
            value != expected for value in declared_by_arms
        ):
            return invalid
    if any(
        record.details.get("model_sha256") != expected
        for record in records
        if _is_replay_contract(record)
    ):
        return invalid
    return ()


def _decoder_adapter_semantic_reasons(
    start: RunStartManifest,
) -> tuple[IntegrityFailureCode, ...]:
    """Bind every arm profile to the run's declared decoder adapter.

    A run start without ``decoder_adapter`` predates adapter selection, so its
    arms may only have used faster-whisper. A NeMo TDT declaration names the
    onnx-asr runtime and the full-precision model manifest every arm used.
    """

    invalid = (IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID,)
    declared = start.details.get("decoder_adapter")
    arm_adapters = {
        definition.parameters.get("decoder_adapter")
        for definition in start.arm_definitions
    }
    if declared is None or (
        isinstance(declared, Mapping)
        and dict(declared) == {"name": DecoderAdapter.FASTER_WHISPER.value}
    ):
        return () if arm_adapters == {None} else invalid
    if (
        not isinstance(declared, Mapping)
        or set(declared) != _NEMO_TDT_IDENTITY_KEYS
        or declared["name"] != _NEMO_TDT
        or declared["model_type"] != _NEMO_TDT_MODEL_TYPE
        or declared["quantization"] is not None
        or declared["onnx_asr_version"] != _NEMO_TDT_ONNX_ASR_VERSION
        or type(declared["onnxruntime_version"]) is not str
        or not declared["onnxruntime_version"]
        or declared["execution_provider"] != _NEMO_TDT_EXECUTION_PROVIDER
        or arm_adapters != {_NEMO_TDT}
        or any(
            definition.kind not in (ArmKind.P0, ArmKind.P1)
            for definition in start.arm_definitions
        )
    ):
        return invalid
    manifest = declared["model_artifact_manifest"]
    try:
        files = manifest["files"] if manifest["kind"] == "tree" else None
        relative_paths = {item["relative_path"] for item in files}
        manifest_sha256 = canonical_sha256(manifest)
    except (KeyError, TypeError, ValueError):
        return invalid
    if not _NEMO_TDT_FP32_FILES <= relative_paths or any(
        definition.parameters.get("model_artifact_sha256") != manifest_sha256
        for definition in start.arm_definitions
    ):
        return invalid
    return ()



def _event_fate_semantic_reasons(
    records: tuple[EvidenceRecord, ...],
    start: RunStartManifest,
) -> tuple[IntegrityFailureCode, ...]:
    """Require exact event/arm fate coverage; every run must declare it."""

    invalid = IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID
    if start.details.get("event_fates_required") is not True:
        return (invalid,)
    raw_event_ids = start.details.get("reference_event_ids")
    if not isinstance(raw_event_ids, tuple) or not raw_event_ids:
        return (invalid,)
    try:
        event_ids = tuple(
            require_uuid4(event_id, "reference_event_id") for event_id in raw_event_ids
        )
    except (TypeError, ValueError):
        return (invalid,)
    if len(event_ids) != len(set(event_ids)):
        return (invalid,)
    requested_arm_ids = set(start.requested_arm_ids)
    expected = {
        (arm_id, event_id) for arm_id in requested_arm_ids for event_id in event_ids
    }
    observed: list[tuple[str, str]] = []
    terminal_seq = {
        record.arm_id: record.seq
        for record in records
        if record.kind is RecordKind.ARM_STATUS and record.arm_id is not None
    }
    for record in records:
        if record.kind is not RecordKind.EVENT_FATE:
            continue
        if (
            record.arm_id not in requested_arm_ids
            or record.artifact_id not in set(event_ids)
            or record.seq >= terminal_seq.get(record.arm_id, -1)
        ):
            return (invalid,)
        observed.append((record.arm_id, record.artifact_id))
    if len(observed) != len(set(observed)) or set(observed) != expected:
        return (invalid,)
    return ()


def _canonical_scan(run_dir: Path, start: RunStartManifest) -> _CanonicalScan:
    raw = _read_stream(run_dir / _CANONICAL_NAME)
    if raw is None:
        return _CanonicalScan(
            records=(),
            summary=None,
            arm_statuses=(),
            reasons=(IntegrityFailureCode.STREAM_MISSING,),
        )
    lines, line_reasons = _complete_lines(raw)
    reasons = list(line_reasons)
    approved = _approved_contracts(start)
    if approved is None:
        reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
    records: list[EvidenceRecord] = []
    records_by_id: dict[str, EvidenceRecord] = {}
    arm_statuses: dict[str, ArmTerminalStatus] = {}
    stream_classification = start.classification
    requested = set(start.requested_arm_ids)

    for index, line in enumerate(lines):
        try:
            wire_value = canonical_json_mapping_from_bytes(
                line,
                f"canonical record {index}",
            )
            record = evidence_record_from_wire(wire_value)
        except (TypeError, ValueError, WireValidationError):
            reasons.append(IntegrityFailureCode.INVALID_OR_TRUNCATED_JSONL)
            continue
        records.append(record)
        try:
            _validate_canonical_payload(
                record.kind,
                record.decision,
                record.details,
                record.observation_status,
                approved=approved,
            )
        except (TypeError, ValueError):
            reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
        if _record_loss_codes(record):
            reasons.append(IntegrityFailureCode.LOSS_REPORTED)
        if record.seq != index:
            reasons.append(IntegrityFailureCode.SEQUENCE_GAP)
        if record.record_id in records_by_id:
            reasons.append(IntegrityFailureCode.RECORD_ID_DUPLICATE)
        if (
            record.run_id != start.run_id
            or record.session_id != start.session_id
            or record.clock_id != start.clock_id
        ):
            reasons.append(IntegrityFailureCode.RECORD_SCOPE_MISMATCH)
        if record.arm_id is not None and record.arm_id not in requested:
            reasons.append(IntegrityFailureCode.UNKNOWN_ARM)
        parent_classes: list[PrivateContentClass] = [start.classification]
        for parent_id in record.parent_ids:
            parent = records_by_id.get(parent_id)
            if parent is None:
                reasons.append(IntegrityFailureCode.PARENT_LINK_INVALID)
            else:
                parent_classes.append(parent.classification)
        if (
            inherited_classification(record.classification, parent_classes)
            is not record.classification
        ):
            reasons.append(IntegrityFailureCode.CLASSIFICATION_MISMATCH)
        records_by_id[record.record_id] = record
        stream_classification = inherited_classification(
            stream_classification,
            (record.classification,),
        )
        if record.kind is RecordKind.ARM_STATUS:
            reason = record.details.get("reason")
            try:
                status = ArmStatus(record.decision)
                terminal = ArmTerminalStatus(
                    arm_id=record.arm_id,
                    status=status,
                    terminal_record_id=record.record_id,
                    reason=reason,
                )
                if (
                    record.stage is not Stage.RUN
                    or record.details.get("status") != status.value
                    or terminal.arm_id in arm_statuses
                ):
                    raise ValueError("invalid arm terminal record")
            except (TypeError, ValueError):
                reasons.append(IntegrityFailureCode.ARM_STATUS_MISMATCH)
            else:
                arm_statuses[terminal.arm_id] = terminal

    reasons.extend(_replay_semantic_reasons(tuple(records)))
    reasons.extend(_capture_replay_semantic_reasons(tuple(records), start))
    reasons.extend(_vad_artifact_semantic_reasons(tuple(records), start))
    reasons.extend(_decoder_adapter_semantic_reasons(start))
    reasons.extend(_decoder_semantic_reasons(tuple(records), start))
    reasons.extend(_event_fate_semantic_reasons(tuple(records), start))

    summary: StreamSummary | None = None
    if len(records) == len(lines):
        try:
            summary = StreamSummary(
                name=StreamName.CANONICAL,
                stream_id=start.canonical_stream_id,
                record_count=len(lines),
                first_seq=records[0].seq if records else None,
                last_seq=records[-1].seq if records else None,
                byte_sha256=hashlib.sha256(raw).hexdigest(),
                classification=stream_classification,
            )
        except (TypeError, ValueError):
            reasons.append(IntegrityFailureCode.STREAM_SUMMARY_MISMATCH)
    return _CanonicalScan(
        records=tuple(records),
        summary=summary,
        arm_statuses=tuple(
            arm_statuses[arm_id]
            for arm_id in start.requested_arm_ids
            if arm_id in arm_statuses
        ),
        reasons=_unique_reasons(reasons),
    )


def _private_scan(
    run_dir: Path,
    start: RunStartManifest,
    canonical_records: tuple[EvidenceRecord, ...],
) -> _PrivateScan:
    expected_bindings: dict[str, tuple[str, TextRole]] = {}
    binding_reasons: list[IntegrityFailureCode] = []
    for record in canonical_records:
        try:
            reference = _text_reference(record)
        except (TypeError, ValueError):
            binding_reasons.append(IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID)
            continue
        if reference is not None:
            expected_bindings[record.record_id] = reference
    if start.private_text_stream_id is None:
        reasons = list(binding_reasons)
        if (run_dir / _PRIVATE_TEXT_NAME).exists():
            reasons.append(IntegrityFailureCode.STREAM_SUMMARY_MISMATCH)
        if expected_bindings:
            reasons.append(IntegrityFailureCode.PRIVATE_TEXT_LINK_MISMATCH)
        return _PrivateScan(
            records=(),
            summary=None,
            reasons=_unique_reasons(reasons),
        )
    raw = _read_stream(run_dir / _PRIVATE_TEXT_NAME)
    if raw is None:
        return _PrivateScan(
            records=(),
            summary=None,
            reasons=(IntegrityFailureCode.STREAM_MISSING,),
        )
    lines, line_reasons = _complete_lines(raw)
    reasons = list(line_reasons)
    reasons.extend(binding_reasons)
    records: list[PrivateTextRecord] = []
    seen_ids: set[str] = set()
    seen_text_ids: set[str] = set()
    bound_canonical_ids: set[str] = set()
    canonical_by_id = {record.record_id: record for record in canonical_records}
    stream_classification = PrivateContentClass.PRIVATE_TEXT
    for index, line in enumerate(lines):
        try:
            wire_value = canonical_json_mapping_from_bytes(
                line,
                f"private text record {index}",
            )
            record = private_text_record_from_wire(wire_value)
        except (TypeError, ValueError, WireValidationError):
            reasons.append(IntegrityFailureCode.INVALID_OR_TRUNCATED_JSONL)
            continue
        records.append(record)
        if record.stream_seq != index:
            reasons.append(IntegrityFailureCode.SEQUENCE_GAP)
        if record.private_record_id in seen_ids:
            reasons.append(IntegrityFailureCode.RECORD_ID_DUPLICATE)
        seen_ids.add(record.private_record_id)
        if record.text_id in seen_text_ids:
            reasons.append(IntegrityFailureCode.PRIVATE_TEXT_LINK_MISMATCH)
        seen_text_ids.add(record.text_id)
        if record.canonical_record_id in bound_canonical_ids:
            reasons.append(IntegrityFailureCode.PRIVATE_TEXT_LINK_MISMATCH)
        bound_canonical_ids.add(record.canonical_record_id)
        if (
            record.run_id != start.run_id
            or record.session_id != start.session_id
            or record.stream_id != start.private_text_stream_id
        ):
            reasons.append(IntegrityFailureCode.RECORD_SCOPE_MISMATCH)
        canonical = canonical_by_id.get(record.canonical_record_id)
        if canonical is None or canonical.arm_id != record.arm_id:
            reasons.append(IntegrityFailureCode.PARENT_LINK_INVALID)
        else:
            expected_binding = expected_bindings.get(record.canonical_record_id)
            if expected_binding != (record.text_id, record.role):
                reasons.append(IntegrityFailureCode.PRIVATE_TEXT_LINK_MISMATCH)
            expected = inherited_classification(
                PrivateContentClass.PRIVATE_TEXT,
                (canonical.classification,),
            )
            if record.classification is not expected:
                reasons.append(IntegrityFailureCode.CLASSIFICATION_MISMATCH)
        stream_classification = inherited_classification(
            stream_classification,
            (record.classification,),
        )
    canonical_ids = {record.record_id for record in canonical_records}
    if seen_ids & canonical_ids:
        reasons.append(IntegrityFailureCode.RECORD_ID_DUPLICATE)
    if set(expected_bindings) != bound_canonical_ids:
        reasons.append(IntegrityFailureCode.PRIVATE_TEXT_LINK_MISMATCH)
    summary: StreamSummary | None = None
    if len(records) == len(lines):
        try:
            summary = StreamSummary(
                name=StreamName.PRIVATE_TEXT,
                stream_id=start.private_text_stream_id,
                record_count=len(lines),
                first_seq=records[0].stream_seq if records else None,
                last_seq=records[-1].stream_seq if records else None,
                byte_sha256=hashlib.sha256(raw).hexdigest(),
                classification=stream_classification,
            )
        except (TypeError, ValueError):
            reasons.append(IntegrityFailureCode.STREAM_SUMMARY_MISMATCH)
    return _PrivateScan(
        records=tuple(records),
        summary=summary,
        reasons=_unique_reasons(reasons),
    )


def _decoder_private_semantic_reasons(
    canonical_records: tuple[EvidenceRecord, ...],
    private_records: tuple[PrivateTextRecord, ...],
    start: RunStartManifest,
) -> tuple[IntegrityFailureCode, ...]:
    """Recompute text-dependent wrapper and P2a decisions from private data."""

    mismatch = IntegrityFailureCode.PRIVATE_TEXT_LINK_MISMATCH
    reasons: list[IntegrityFailureCode] = []
    approved = _approved_contracts(start)
    try:
        no_speech_only_threshold = _no_speech_only_threshold(start)
    except ValueError:
        # _decoder_semantic_reasons reports the malformed setting.
        no_speech_only_threshold = None
    private_by_canonical = {
        record.canonical_record_id: record for record in private_records
    }
    calls = {
        record.record_id: record
        for record in canonical_records
        if record.kind is RecordKind.DECODER_CALL
        and _decoder_payload_valid(record, approved)
    }
    raw_by_call: dict[str, list[EvidenceRecord]] = {call_id: [] for call_id in calls}
    for record in canonical_records:
        if (
            record.kind is RecordKind.RAW_HYPOTHESIS
            and set(record.details)
            in (_RAW_HYPOTHESIS_KEYS, _NEMO_TDT_RAW_HYPOTHESIS_KEYS)
            and _decoder_payload_valid(record, approved)
            and len(record.parent_ids) == 1
            and record.parent_ids[0] in calls
            and _is_nemo_tdt_call(calls[record.parent_ids[0]])
            == (set(record.details) == _NEMO_TDT_RAW_HYPOTHESIS_KEYS)
        ):
            raw_by_call[record.parent_ids[0]].append(record)
    wrappers_by_parent = _wrapper_positions_by_parent(canonical_records)

    greedy_texts_by_call: dict[str, tuple[tuple[str, ...], bool]] = {}
    for call_id, call in calls.items():
        raw_records = sorted(
            raw_by_call[call_id],
            key=lambda record: record.details["segment_ordinal"],
        )
        try:
            raw_texts = tuple(
                private_by_canonical[record.record_id].text for record in raw_records
            )
        except KeyError:
            reasons.append(mismatch)
            continue
        if _is_nemo_tdt_call(call):
            # The candidate's only gate is text-derived: re-derive it from the
            # private text, and require the ratio the record claims.
            gate = text_only_wrapper_gate("".join(raw_texts))
            if any(
                record.details["nemo_tdt"]["compression_ratio"]
                != gate.compression_ratio
                for record in raw_records
            ):
                reasons.append(mismatch)
                continue
            greedy_texts_by_call[call_id] = (raw_texts, False)
            assembled = gate.assembled_text
            compression_rejected = gate.compression_rejected
        else:
            tail = stochastic_tail_start(
                record.details["temperature"] for record in raw_records
            )
            greedy_texts_by_call[call_id] = (raw_texts[:tail], tail < len(raw_texts))
            kept = tuple(
                (record, text)
                for record, text in zip(raw_records, raw_texts)
                if not _whisper_confidence_rejected(
                    record.details["no_speech_prob"],
                    record.details["avg_logprob"],
                    no_speech_only_threshold,
                )
            )
            assembled = "".join(text for _, text in kept).strip()
            compression_rejected = bool(
                len(assembled) >= 30
                and kept
                and max(record.details["compression_ratio"] for record, _ in kept)
                > 2.4
            )
        expected_decision = (
            DiagnosticCode.WRAPPER_COMPRESSION_REJECT.value
            if compression_rejected
            else TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value
            if assembled
            else DiagnosticCode.WRAPPER_EMPTY.value
        )
        raw_ids = {record.record_id for record in raw_records}
        confidence_ids = {
            record.record_id
            for record in _wrapper_decisions_under(
                canonical_records,
                wrappers_by_parent,
                raw_ids,
            )
            if record.decision == DiagnosticCode.WRAPPER_CONFIDENCE_GATE.value
            and len(record.parent_ids) == 1
        }
        related_ids = {call_id, *raw_ids, *confidence_ids}
        terminals = tuple(
            record
            for record in _wrapper_decisions_under(
                canonical_records,
                wrappers_by_parent,
                related_ids,
            )
            if record.decision
            in {
                DiagnosticCode.WRAPPER_COMPRESSION_REJECT.value,
                DiagnosticCode.WRAPPER_EMPTY.value,
                TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value,
            }
            and all(parent_id in related_ids for parent_id in record.parent_ids)
        )
        if len(terminals) != 1 or terminals[0].decision != expected_decision:
            reasons.append(mismatch)
            continue
        terminal = terminals[0]
        if expected_decision == TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value:
            private = private_by_canonical.get(terminal.record_id)
            if private is None or private.text != assembled:
                reasons.append(mismatch)

    for call_id, call in calls.items():
        if call.details["call_mode"] != ArmKind.P2A.value:
            continue
        parent_id = call.details["parent_call_record_id"]
        if (
            call_id not in greedy_texts_by_call
            or parent_id not in greedy_texts_by_call
            or greedy_texts_by_call[call_id] != greedy_texts_by_call[parent_id]
        ):
            reasons.append(mismatch)

    return _unique_reasons(reasons)


def _compare_summaries(
    declared: StreamSummary | None,
    computed: StreamSummary | None,
    reasons: list[IntegrityFailureCode],
) -> None:
    if declared is None or computed is None:
        reasons.append(IntegrityFailureCode.STREAM_SUMMARY_MISMATCH)
        return
    if declared.byte_sha256 != computed.byte_sha256:
        reasons.append(IntegrityFailureCode.STREAM_HASH_MISMATCH)
    if (
        declared.name != computed.name
        or declared.stream_id != computed.stream_id
        or declared.record_count != computed.record_count
        or declared.first_seq != computed.first_seq
        or declared.last_seq != computed.last_seq
        or declared.classification != computed.classification
    ):
        reasons.append(IntegrityFailureCode.STREAM_SUMMARY_MISMATCH)


def _derive_status(
    reasons: tuple[IntegrityFailureCode, ...],
    arm_statuses: tuple[ArmTerminalStatus, ...],
) -> RunStatus:
    if not reasons and all(item.status is ArmStatus.COMPLETE for item in arm_statuses):
        return RunStatus.COMPLETE
    if any(item.status is ArmStatus.FAILED for item in arm_statuses):
        return RunStatus.FAILED
    if any(item.status is ArmStatus.CANCELLED for item in arm_statuses):
        return RunStatus.CANCELLED
    return RunStatus.INCOMPLETE


class LedgerWriter(EvidenceSink):
    def __init__(
        self,
        run_dir: Path,
        start: RunStartManifest,
        start_bytes: bytes,
        canonical: _AppendStream,
        private_text: _AppendStream | None,
        id_factory: Callable[[], str],
        monotonic_ns: Callable[[], int],
    ) -> None:
        self.run_dir = run_dir
        self.start = start
        self._start_bytes = start_bytes
        self._canonical = canonical
        self._private_text = private_text
        self._id_factory = id_factory
        self._monotonic_ns = monotonic_ns
        self._last_monotonic_ns: int | None = None
        self._records: dict[str, EvidenceRecord] = {}
        self._private_links: dict[str, str] = {}
        self._arm_statuses: dict[str, ArmTerminalStatus] = {}
        self._loss_counts = {code: 0 for code in LossCode}
        self._issued_ids = {
            start.run_id,
            start.session_id,
            start.canonical_stream_id,
            *start.requested_arm_ids,
        }
        if start.private_text_stream_id is not None:
            self._issued_ids.add(start.private_text_stream_id)
        self._poisoned = False
        self._closing = False
        self._finalized: RunFinalManifest | None = None
        self._approved_contracts = _approved_contracts(start)

    @classmethod
    def create(
        cls,
        run_dir: Path,
        start: RunStartManifest,
        *,
        boundary: PrivateBoundary,
        id_factory: Callable[[], str] = new_opaque_id,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> "LedgerWriter":
        if not isinstance(run_dir, Path):
            raise TypeError("run_dir must be Path")
        if not isinstance(start, RunStartManifest):
            raise TypeError("start must be RunStartManifest")
        run_dir = require_private_path(run_dir, boundary)
        run_dir.mkdir(parents=True, exist_ok=False)
        start_bytes = canonical_json_bytes(run_start_to_wire(start), newline=True)
        atomic_write_bytes(run_dir / _RUN_START_NAME, start_bytes)
        canonical: _AppendStream | None = None
        private_text: _AppendStream | None = None
        try:
            canonical = _AppendStream(
                run_dir / _CANONICAL_NAME,
                StreamName.CANONICAL,
                start.canonical_stream_id,
                start.classification,
            )
            if start.private_text_stream_id is not None:
                private_text = _AppendStream(
                    run_dir / _PRIVATE_TEXT_NAME,
                    StreamName.PRIVATE_TEXT,
                    start.private_text_stream_id,
                    PrivateContentClass.PRIVATE_TEXT,
                )
            return cls(
                run_dir,
                start,
                start_bytes,
                canonical,
                private_text,
                id_factory,
                monotonic_ns,
            )
        except BaseException:
            if canonical is not None:
                canonical.close_without_fsync()
            if private_text is not None:
                private_text.close_without_fsync()
            raise

    def _ensure_writable(self) -> None:
        if self._finalized is not None or self._closing:
            raise RuntimeError("ledger writer is finalized")
        if self._poisoned:
            raise RuntimeError("ledger writer is poisoned by an earlier write")

    def _new_id(self, field_name: str) -> str:
        value = require_uuid4(self._id_factory(), field_name)
        if value in self._issued_ids:
            raise ValueError(f"{field_name} must be globally unique")
        self._issued_ids.add(value)
        return value

    def _next_time(self) -> int:
        value = self._monotonic_ns()
        if type(value) is not int or value < 0:
            raise TypeError("monotonic clock must return a nonnegative integer")
        if self._last_monotonic_ns is not None and value < self._last_monotonic_ns:
            raise ValueError("monotonic clock moved backwards")
        self._last_monotonic_ns = value
        return value

    def _record_write_failure(self) -> None:
        self._loss_counts[LossCode.RECORD_LOST] += 1
        self._poisoned = True

    def _emit(
        self,
        draft: EvidenceRecordDraft,
        *,
        allow_arm_status: bool,
    ) -> EvidenceRecord:
        self._ensure_writable()
        if not isinstance(draft, EvidenceRecordDraft):
            raise TypeError("draft must be EvidenceRecordDraft")
        if draft.kind in {RecordKind.ARM_STATUS, RecordKind.RUN_STATUS} and not (
            allow_arm_status and draft.kind is RecordKind.ARM_STATUS
        ):
            raise ValueError("status records must use the status writer methods")
        if draft.arm_id is not None:
            if draft.arm_id not in self.start.requested_arm_ids:
                raise ValueError("draft arm was not requested for this run")
            if draft.arm_id in self._arm_statuses:
                raise ValueError("cannot emit after an arm terminal status")
        _validate_canonical_payload(
            draft.kind,
            draft.decision,
            draft.details,
            draft.observation_status,
            approved=self._approved_contracts,
        )
        if _text_reference(draft) is not None and self._private_text is None:
            raise PermissionError("canonical text references require a private stream")
        parent_classes: list[PrivateContentClass] = [self.start.classification]
        for parent_id in draft.parent_ids:
            parent = self._records.get(parent_id)
            if parent is None:
                raise ValueError("parent_ids must name earlier canonical records")
            parent_classes.append(parent.classification)
        classification = inherited_classification(
            draft.classification,
            parent_classes,
        )
        record = EvidenceRecord(
            schema_version=SCHEMA_VERSION,
            record_id=self._new_id("record_id"),
            run_id=self.start.run_id,
            arm_id=draft.arm_id,
            session_id=self.start.session_id,
            seq=self._canonical.count,
            observed_monotonic_ns=self._next_time(),
            clock_id=self.start.clock_id,
            stage=draft.stage,
            kind=draft.kind,
            artifact_id=draft.artifact_id,
            parent_ids=draft.parent_ids,
            source_spans=draft.source_spans,
            observation_status=draft.observation_status,
            decision=draft.decision,
            diagnostics=draft.diagnostics,
            classification=classification,
            details=draft.details,
        )
        payload = canonical_json_bytes(
            evidence_record_to_wire(record),
            newline=True,
        )
        try:
            self._canonical.append(payload, record.seq, record.classification)
        except OSError:
            self._record_write_failure()
            raise
        self._records[record.record_id] = record
        for code in _record_loss_codes(record):
            self._loss_counts[code] += 1
        return record

    def emit(self, draft: EvidenceRecordDraft) -> EvidenceRecord:
        return self._emit(draft, allow_arm_status=False)

    def emit_private_text(
        self,
        *,
        canonical_record_id: str,
        arm_id: str,
        text_id: str,
        role: TextRole,
        text: str,
    ) -> str:
        self._ensure_writable()
        if self._private_text is None or self.start.private_text_stream_id is None:
            raise PermissionError("private text stream is not enabled")
        canonical = self._records.get(canonical_record_id)
        if canonical is None:
            raise ValueError("canonical_record_id is not an earlier record")
        if canonical.arm_id != arm_id:
            raise ValueError("private text arm does not match canonical record")
        if arm_id in self._arm_statuses:
            raise ValueError("cannot emit private text after an arm terminal status")
        expected_binding = _text_reference(canonical)
        if expected_binding != (text_id, role):
            raise ValueError(
                "private text identity or role does not match canonical record"
            )
        if canonical_record_id in self._private_links:
            raise ValueError("canonical record already has a private text binding")
        classification = inherited_classification(
            PrivateContentClass.PRIVATE_TEXT,
            (canonical.classification,),
        )
        record = PrivateTextRecord(
            schema_version=SCHEMA_VERSION,
            private_record_id=self._new_id("private_record_id"),
            run_id=self.start.run_id,
            session_id=self.start.session_id,
            stream_id=self.start.private_text_stream_id,
            stream_seq=self._private_text.count,
            canonical_record_id=canonical_record_id,
            arm_id=arm_id,
            text_id=text_id,
            role=role,
            classification=classification,
            text=text,
        )
        payload = canonical_json_bytes(
            private_text_record_to_wire(record),
            newline=True,
        )
        try:
            self._private_text.append(
                payload,
                record.stream_seq,
                record.classification,
            )
        except OSError:
            self._record_write_failure()
            raise
        self._private_links[canonical_record_id] = record.private_record_id
        return record.private_record_id

    def finish_arm(
        self,
        arm_id: str,
        status: ArmStatus,
        *,
        reason: str | None = None,
        exception_class: str | None = None,
    ) -> EvidenceRecord:
        """Write an arm's terminal status.

        ``exception_class`` names the exception that stopped a noncomplete arm.
        """

        self._ensure_writable()
        if arm_id not in self.start.requested_arm_ids:
            raise ValueError("arm was not requested for this run")
        if arm_id in self._arm_statuses:
            raise ValueError("arm already has a terminal status")
        if not isinstance(status, ArmStatus) or status not in _TERMINAL_ARM_STATUSES:
            raise ValueError("status must be a terminal ArmStatus")
        if status is ArmStatus.COMPLETE:
            if reason is not None or exception_class is not None:
                raise ValueError("complete arm status must not have a reason")
            missing_text = tuple(
                record.record_id
                for record in self._records.values()
                if record.arm_id == arm_id
                and _text_reference(record) is not None
                and record.record_id not in self._private_links
            )
            if missing_text:
                raise ValueError(
                    "complete arm status requires every text reference to be bound"
                )
        elif type(reason) is not str or reason not in _ARM_REASON_VALUES:
            raise ValueError(
                "noncomplete arm status requires a closed-vocabulary reason"
            )
        details: dict[str, str] = {"status": status.value}
        if reason is not None:
            details["reason"] = reason
        if exception_class is not None:
            details["exception_class"] = _require_exception_class(exception_class)
        record = self._emit(
            EvidenceRecordDraft(
                arm_id=arm_id,
                stage=Stage.RUN,
                kind=RecordKind.ARM_STATUS,
                artifact_id=arm_id,
                observation_status=ObservationStatus.OBSERVED,
                decision=status.value,
                classification=self.start.classification,
                details=details,
            ),
            allow_arm_status=True,
        )
        self._arm_statuses[arm_id] = ArmTerminalStatus(
            arm_id=arm_id,
            status=status,
            terminal_record_id=record.record_id,
            reason=reason,
        )
        return record

    def finalize(self) -> RunFinalManifest:
        if self._finalized is not None or self._closing:
            raise RuntimeError("ledger writer is already finalized")
        self._closing = True
        reasons: list[IntegrityFailureCode] = []
        streams = [self._canonical]
        if self._private_text is not None:
            streams.append(self._private_text)
        intended = tuple(stream.intended_summary() for stream in streams)
        for stream in streams:
            try:
                stream.finish()
            except OSError:
                self._loss_counts[LossCode.RECORD_LOST] += 1
                reasons.append(IntegrityFailureCode.STREAM_WRITE_MISMATCH)
                stream.close_without_fsync()

        canonical_scan = _canonical_scan(self.run_dir, self.start)
        private_scan = _private_scan(
            self.run_dir,
            self.start,
            canonical_scan.records,
        )
        reasons.extend(canonical_scan.reasons)
        reasons.extend(private_scan.reasons)
        reasons.extend(
            _decoder_private_semantic_reasons(
                canonical_scan.records,
                private_scan.records,
                self.start,
            )
        )
        computed_by_name = {
            summary.name: summary
            for summary in (canonical_scan.summary, private_scan.summary)
            if summary is not None
        }
        for summary in intended:
            before = len(reasons)
            _compare_summaries(
                summary,
                computed_by_name.get(summary.name),
                reasons,
            )
            if len(reasons) != before:
                reasons.append(IntegrityFailureCode.STREAM_WRITE_MISMATCH)
        arm_statuses = tuple(
            self._arm_statuses[arm_id]
            for arm_id in self.start.requested_arm_ids
            if arm_id in self._arm_statuses
        )
        if canonical_scan.arm_statuses != arm_statuses:
            reasons.append(IntegrityFailureCode.ARM_STATUS_MISMATCH)
        missing = set(self.start.requested_arm_ids) - set(self._arm_statuses)
        if missing:
            reasons.append(IntegrityFailureCode.ARM_TERMINAL_STATUS_MISSING)
        if any(
            item.status is not ArmStatus.COMPLETE
            for item in self._arm_statuses.values()
        ):
            reasons.append(IntegrityFailureCode.ARM_STATUS_MISMATCH)
        if any(self._loss_counts.values()):
            reasons.append(IntegrityFailureCode.LOSS_REPORTED)
        final_reasons = _unique_reasons(reasons)
        status = _derive_status(final_reasons, arm_statuses)
        classification = self.start.classification
        for summary in intended:
            classification = inherited_classification(
                classification,
                (summary.classification,),
            )
        final = RunFinalManifest(
            schema_version=SCHEMA_VERSION,
            run_id=self.start.run_id,
            session_id=self.start.session_id,
            run_start_sha256=hashlib.sha256(self._start_bytes).hexdigest(),
            status=status,
            streams=intended,
            arm_statuses=arm_statuses,
            loss_counts=self._loss_counts,
            incomplete_reasons=final_reasons,
            classification=classification,
            finalized_monotonic_ns=self._next_time(),
        )
        payload = canonical_json_bytes(run_final_to_wire(final), newline=True)
        atomic_write_bytes(self.run_dir / _RUN_FINAL_NAME, payload)
        self._finalized = final
        return final


def _load_start(
    run_dir: Path,
) -> tuple[RunStartManifest | None, bytes | None, tuple[IntegrityFailureCode, ...]]:
    raw = _read_stream(run_dir / _RUN_START_NAME)
    if raw is None:
        return None, None, (IntegrityFailureCode.START_MANIFEST_MISSING,)
    try:
        value = canonical_json_mapping_from_bytes(raw, "run start manifest")
        return run_start_from_wire(value), raw, ()
    except (TypeError, ValueError, WireValidationError):
        return None, raw, (IntegrityFailureCode.START_MANIFEST_INVALID,)


def _load_final(
    run_dir: Path,
) -> tuple[RunFinalManifest | None, tuple[IntegrityFailureCode, ...]]:
    raw = _read_stream(run_dir / _RUN_FINAL_NAME)
    if raw is None:
        return None, (IntegrityFailureCode.FINAL_MANIFEST_MISSING,)
    try:
        value = canonical_json_mapping_from_bytes(raw, "run final manifest")
        return run_final_from_wire(value), ()
    except (TypeError, ValueError, WireValidationError):
        return None, (IntegrityFailureCode.FINAL_MANIFEST_INVALID,)


def iter_canonical_records(run_dir: Path) -> Iterator[EvidenceRecord]:
    if not isinstance(run_dir, Path):
        raise TypeError("run_dir must be Path")
    start, _, start_reasons = _load_start(run_dir)
    if start is None:
        codes = ",".join(reason.value for reason in start_reasons)
        raise ValueError(f"cannot read canonical ledger: {codes}")
    scan = _canonical_scan(run_dir, start)
    if scan.reasons:
        codes = ",".join(reason.value for reason in scan.reasons)
        raise ValueError(f"canonical ledger integrity failure: {codes}")
    yield from scan.records


def iter_private_text_records(run_dir: Path) -> Iterator[PrivateTextRecord]:
    """Yield a run's private text records once both streams scan cleanly."""

    if not isinstance(run_dir, Path):
        raise TypeError("run_dir must be Path")
    start, _, start_reasons = _load_start(run_dir)
    if start is None:
        codes = ",".join(reason.value for reason in start_reasons)
        raise ValueError(f"cannot read private text stream: {codes}")
    canonical_scan = _canonical_scan(run_dir, start)
    scan = _private_scan(run_dir, start, canonical_scan.records)
    reasons = _unique_reasons((*canonical_scan.reasons, *scan.reasons))
    if reasons:
        codes = ",".join(reason.value for reason in reasons)
        raise ValueError(f"private text stream integrity failure: {codes}")
    yield from scan.records


def verify_run(run_dir: Path) -> IntegrityReport:
    if not isinstance(run_dir, Path):
        raise TypeError("run_dir must be Path")
    reasons: list[IntegrityFailureCode] = []
    start, start_raw, start_reasons = _load_start(run_dir)
    reasons.extend(start_reasons)
    final, final_reasons = _load_final(run_dir)
    reasons.extend(final_reasons)
    if start is None:
        return IntegrityReport(
            run_id=None,
            complete=False,
            declared_status=final.status if final is not None else None,
            computed_streams=(),
            computed_arm_statuses=(),
            reasons=_unique_reasons(reasons),
        )

    canonical_scan = _canonical_scan(run_dir, start)
    private_scan = _private_scan(run_dir, start, canonical_scan.records)
    reasons.extend(canonical_scan.reasons)
    reasons.extend(private_scan.reasons)
    reasons.extend(
        _decoder_private_semantic_reasons(
            canonical_scan.records,
            private_scan.records,
            start,
        )
    )
    computed_streams = tuple(
        summary
        for summary in (canonical_scan.summary, private_scan.summary)
        if summary is not None
    )
    computed_arm_statuses = canonical_scan.arm_statuses
    missing = set(start.requested_arm_ids) - {
        item.arm_id for item in computed_arm_statuses
    }
    if missing:
        reasons.append(IntegrityFailureCode.ARM_TERMINAL_STATUS_MISSING)
    if any(item.status is not ArmStatus.COMPLETE for item in computed_arm_statuses):
        reasons.append(IntegrityFailureCode.ARM_STATUS_MISMATCH)

    if final is not None:
        if (
            final.run_id != start.run_id
            or final.session_id != start.session_id
            or start_raw is None
            or final.run_start_sha256 != hashlib.sha256(start_raw).hexdigest()
        ):
            reasons.append(IntegrityFailureCode.FINAL_STATUS_MISMATCH)
        declared_streams = {summary.name: summary for summary in final.streams}
        computed_by_name = {summary.name: summary for summary in computed_streams}
        expected_names = {StreamName.CANONICAL}
        if start.private_text_stream_id is not None:
            expected_names.add(StreamName.PRIVATE_TEXT)
        if set(declared_streams) != expected_names:
            reasons.append(IntegrityFailureCode.STREAM_SUMMARY_MISMATCH)
        for name in expected_names:
            _compare_summaries(
                declared_streams.get(name),
                computed_by_name.get(name),
                reasons,
            )
        if final.arm_statuses != computed_arm_statuses:
            reasons.append(IntegrityFailureCode.ARM_STATUS_MISMATCH)
        if any(final.loss_counts.values()):
            reasons.append(IntegrityFailureCode.LOSS_REPORTED)
        computed_classification = start.classification
        for summary in computed_streams:
            computed_classification = inherited_classification(
                computed_classification,
                (summary.classification,),
            )
        if final.classification is not computed_classification:
            reasons.append(IntegrityFailureCode.CLASSIFICATION_MISMATCH)
        operational_reasons = _unique_reasons(reasons)
        for reason in final.incomplete_reasons:
            if reason not in reasons:
                reasons.append(reason)
        expected_status = _derive_status(
            _unique_reasons(reasons),
            computed_arm_statuses,
        )
        if final.status is not expected_status or set(final.incomplete_reasons) != set(
            operational_reasons
        ):
            reasons.append(IntegrityFailureCode.FINAL_STATUS_MISMATCH)
    all_reasons = _unique_reasons(reasons)
    complete = (
        final is not None and final.status is RunStatus.COMPLETE and not all_reasons
    )
    return IntegrityReport(
        run_id=start.run_id,
        complete=complete,
        declared_status=final.status if final is not None else None,
        computed_streams=computed_streams,
        computed_arm_statuses=computed_arm_statuses,
        reasons=all_reasons,
    )


def require_complete_run(run_dir: Path) -> None:
    report = verify_run(run_dir)
    if not report.complete:
        raise IncompleteRunError(report)
