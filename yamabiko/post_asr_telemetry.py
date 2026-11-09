"""Privacy-safe lineage telemetry for text that survived ASR.

The tracker deliberately retains only an opaque run coordinate and monotonic
counters.  It never receives transcript text, audio, prompts, content hashes,
or exception details.  Lineage is emitted as fixed-arity log records so a
merge has two ordinary parent edges rather than a variable-shaped payload.
"""
from __future__ import annotations

import logging
import re
import secrets
from dataclasses import dataclass
from enum import Enum


class ArtifactKind(str, Enum):
    ASR_WINDOW = "asr_window"
    LEGACY_UNATTRIBUTED = "legacy_unattributed"
    SENTENCE = "sentence"
    TAIL = "tail"
    MERGED = "merged"


class Stage(str, Enum):
    DEDUP = "dedup"
    LANGUAGE_POLICY = "language_policy"
    SENTENCE_SPLIT = "sentence_split"
    TAIL = "tail"
    TRANSLATION = "translation"
    DISPLAY = "display"


class Decision(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    UNCHANGED = "unchanged"
    REDUCED = "reduced"
    LANGUAGE_POLICY_ACCEPTED = "language_policy_accepted"
    TAIL_BUFFERED = "tail_buffered"
    TAIL_CONSUMED = "tail_consumed"
    TAIL_FORCE_FLUSHED = "tail_force_flushed"
    TRANSLATION_COMPLETED = "translation_completed"


class Fate(str, Enum):
    DEDUP_EMPTY = "dedup_empty"
    LANGUAGE_POLICY_REJECTED = "language_policy_rejected"
    SENTENCE_SPLIT_EMPTY = "sentence_split_empty"
    DISPLAY_POLICY_INTERJECTION = "display_policy_interjection"
    TAIL_EXPIRED = "tail_expired"
    TAIL_DISCONTINUITY_CLEARED = "tail_discontinuity_cleared"
    PIPELINE_STAGE_ERROR = "pipeline_stage_error"
    TRANSLATION_ERROR = "translation_error"
    TRANSLATION_EMPTY = "translation_empty"
    DISPLAY_SEND_COMPLETED = "display_send_completed"
    DISPLAY_NO_CLIENTS = "display_no_clients"
    DISPLAY_ALL_SENDS_FAILED = "display_all_sends_failed"
    DISPLAY_NOT_CONFIGURED = "display_not_configured"
    DISPLAY_RECEIPT_UNAVAILABLE = "display_receipt_unavailable"
    DISPLAY_BROADCAST_ERROR = "display_broadcast_error"


class Phase(str, Enum):
    DEDUP = "dedup"
    LANGUAGE_POLICY = "language_policy"
    SENTENCE_SPLIT = "sentence_split"
    ROMANIZATION_GLOSSARY = "romanization_glossary"
    SERIES_CONTEXT = "series_context"
    POST_TRANSLATION_SCAFFOLDING = "post_translation_scaffolding"
    DISPLAY = "display"


@dataclass(frozen=True)
class ArtifactRef:
    artifact_seq: int

    def __post_init__(self) -> None:
        if type(self.artifact_seq) is not int or self.artifact_seq < 0:
            raise ValueError("artifact_seq must be a non-negative integer")


class PostASRTelemetry:
    """Emit content-free post-ASR lineage without retaining the DAG."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        candidate_run_id = run_id if run_id is not None else secrets.token_hex(6)
        if (
            not isinstance(candidate_run_id, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,32}", candidate_run_id) is None
        ):
            # Never echo the rejected value: it may itself contain transcript
            # text or log-control characters.
            raise ValueError("run_id must be a short opaque ASCII token")
        self.run_id = candidate_run_id
        self._logger = logger if logger is not None else logging.getLogger("yamabiko")
        self._next_window_seq = 0
        self._next_artifact_seq = 0

    def new_window(self) -> int:
        window_seq = self._next_window_seq
        self._next_window_seq += 1
        return window_seq

    def _new_artifact(self) -> ArtifactRef:
        ref = ArtifactRef(self._next_artifact_seq)
        self._next_artifact_seq += 1
        return ref

    def root(self, window_seq: int) -> ArtifactRef:
        if type(window_seq) is not int or window_seq < 0:
            raise ValueError("window_seq must be a non-negative integer")
        ref = self._new_artifact()
        self._logger.info(
            "post_asr root: run_id=%s artifact_seq=%d kind=%s window_seq=%d",
            self.run_id,
            ref.artifact_seq,
            ArtifactKind.ASR_WINDOW.value,
            window_seq,
        )
        return ref

    def adopt_legacy(self) -> ArtifactRef:
        ref = self._new_artifact()
        self._logger.info(
            "post_asr root: run_id=%s artifact_seq=%d kind=%s window_seq=None",
            self.run_id,
            ref.artifact_seq,
            ArtifactKind.LEGACY_UNATTRIBUTED.value,
        )
        return ref

    def child(
        self,
        parent: ArtifactRef,
        kind: ArtifactKind,
        ordinal: int = 0,
    ) -> ArtifactRef:
        if not isinstance(parent, ArtifactRef):
            raise TypeError("parent must be an ArtifactRef")
        if not isinstance(kind, ArtifactKind):
            raise TypeError("kind must be an ArtifactKind")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        ref = self._new_artifact()
        self._edge(ref, parent, kind, ordinal)
        return ref

    def merge(self, left: ArtifactRef, right: ArtifactRef) -> ArtifactRef:
        if not isinstance(left, ArtifactRef) or not isinstance(right, ArtifactRef):
            raise TypeError("merge parents must be ArtifactRef values")
        ref = self._new_artifact()
        self._edge(ref, left, ArtifactKind.MERGED, 0)
        self._edge(ref, right, ArtifactKind.MERGED, 1)
        return ref

    def _edge(
        self,
        child: ArtifactRef,
        parent: ArtifactRef,
        kind: ArtifactKind,
        ordinal: int,
    ) -> None:
        self._logger.info(
            "post_asr edge: run_id=%s artifact_seq=%d parent_artifact_seq=%d "
            "kind=%s ordinal=%d",
            self.run_id,
            child.artifact_seq,
            parent.artifact_seq,
            kind.value,
            ordinal,
        )

    def decision(
        self,
        ref: ArtifactRef,
        stage: Stage,
        decision: Decision,
    ) -> None:
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if not isinstance(stage, Stage) or not isinstance(decision, Decision):
            raise TypeError("stage and decision must use closed telemetry enums")
        self._logger.info(
            "post_asr decision: run_id=%s artifact_seq=%d stage=%s decision=%s",
            self.run_id,
            ref.artifact_seq,
            stage.value,
            decision.value,
        )

    def terminal(
        self,
        ref: ArtifactRef,
        fate: Fate,
        phase: Phase | None = None,
        **numeric_counts: int,
    ) -> None:
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if not isinstance(fate, Fate):
            raise TypeError("fate must use the closed telemetry enum")
        if phase is not None and not isinstance(phase, Phase):
            raise TypeError("phase must use the closed telemetry enum")
        allowed_counts = {"attempted", "accepted", "failed"}
        supplied_counts = set(numeric_counts)
        if supplied_counts and supplied_counts != allowed_counts:
            raise ValueError("post-ASR client counts must be supplied together")
        if any(
            type(value) is not int or value < 0
            for value in numeric_counts.values()
        ):
            raise ValueError(
                "post-ASR numeric counts must be non-negative integers",
            )
        self._logger.info(
            "post_asr terminal: run_id=%s artifact_seq=%d fate=%s phase=%s "
            "attempted=%s accepted=%s failed=%s",
            self.run_id,
            ref.artifact_seq,
            fate.value,
            phase.value if phase is not None else "None",
            numeric_counts.get("attempted"),
            numeric_counts.get("accepted"),
            numeric_counts.get("failed"),
        )
