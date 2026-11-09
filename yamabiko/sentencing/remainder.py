"""Epoch-owned unfinished Japanese sentence state."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral

from yamabiko.post_asr_telemetry import ArtifactRef


def _epoch(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("epoch must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError("epoch must be a non-negative integer")
    return normalized


@dataclass(frozen=True)
class TailPrefix:
    """One read of the unfinished tail on behalf of an incoming segment.

    ``artifact_ref`` is the lineage reference of the tail this read returned
    (to be consumed) or discarded as expired; None when nothing was buffered
    or telemetry was off.
    """

    text: str = field(repr=False)
    expired: bool
    artifact_ref: ArtifactRef | None


@dataclass(slots=True)
class PendingRemainderState:
    """Own at most one unfinished tail for exactly one pause epoch.

    Transcript content is deliberately excluded from ``repr`` because this
    shared object can appear in debug diagnostics. All mutators reject stale
    epochs without exposing the rejected text in exceptions or logs. The tail
    carries its own post-ASR lineage reference, so whichever path ends the
    tail (consume, expiry, pause, capture gap) can record its fate.
    """

    active_epoch: int = 0
    text: str = field(default="", repr=False)
    buffered_at: float = 0.0
    artifact_ref: ArtifactRef | None = None

    def __post_init__(self) -> None:
        self.active_epoch = _epoch(self.active_epoch)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        self.buffered_at = float(self.buffered_at)
        if not math.isfinite(self.buffered_at):
            raise ValueError("buffered_at must be finite")
        if self.artifact_ref is not None and not isinstance(
            self.artifact_ref, ArtifactRef,
        ):
            raise TypeError("artifact_ref must be an ArtifactRef or None")

    def _clear(self) -> None:
        self.text = ""
        self.buffered_at = 0.0
        self.artifact_ref = None

    def _abandon(self) -> ArtifactRef | None:
        ref = self.artifact_ref if self.text else None
        self._clear()
        return ref

    def advance_epoch(self, epoch: int) -> ArtifactRef | None:
        """Move forward and abandon unfinished text; ignore stale repeats.

        Returns the abandoned tail's reference when text was buffered.
        """

        normalized = _epoch(epoch)
        if normalized <= self.active_epoch:
            return None
        self.active_epoch = normalized
        return self._abandon()

    def abandon_for(self, epoch: int) -> ArtifactRef | None:
        """Abandon the tail after a capture gap in the consumer's epoch.

        Returns the abandoned tail's reference when text was buffered.
        """

        if _epoch(epoch) != self.active_epoch:
            return None
        return self._abandon()

    def prefix_for(
        self,
        epoch: int,
        now: float,
        max_age_s: float,
    ) -> TailPrefix:
        """Return the same-epoch prefix and whether it expired on this read."""

        normalized = _epoch(epoch)
        current_time = float(now)
        maximum_age = float(max_age_s)
        if not math.isfinite(current_time) or not math.isfinite(maximum_age):
            raise ValueError("time values must be finite")
        if maximum_age < 0:
            raise ValueError("max_age_s must be non-negative")
        if normalized != self.active_epoch or not self.text:
            return TailPrefix("", False, None)
        if current_time - self.buffered_at > maximum_age:
            return TailPrefix("", True, self._abandon())
        return TailPrefix(self.text, False, self.artifact_ref)

    def clear_for(self, epoch: int) -> None:
        """Clear only when the consumer still owns the active epoch."""

        if _epoch(epoch) == self.active_epoch:
            self._clear()

    def store_for(
        self,
        text: str,
        epoch: int,
        now: float,
        artifact_ref: ArtifactRef | None = None,
    ) -> bool:
        """Store a tail only for the active epoch; reject stale work."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = _epoch(epoch)
        buffered_at = float(now)
        if not math.isfinite(buffered_at):
            raise ValueError("now must be finite")
        if artifact_ref is not None and not isinstance(artifact_ref, ArtifactRef):
            raise TypeError("artifact_ref must be an ArtifactRef or None")
        if normalized != self.active_epoch:
            return False
        self.text = text
        self.buffered_at = buffered_at
        self.artifact_ref = artifact_ref
        return True
