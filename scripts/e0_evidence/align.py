"""Deterministic character and temporal alignment for E0 scoring."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from ._common import require_sha256, require_uuid4
from .schema import SampleSpan
from .wire import canonical_sha256


class EditKind(str, Enum):
    MATCH = "match"
    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"


class AlignmentStatus(str, Enum):
    RESOLVED = "resolved"
    ALIGNMENT_UNRESOLVED = "alignment_unresolved"


def _nonempty(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{field_name} must be a non-empty string")
    return value


def _plain_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def _freeze_strings(values: Iterable[object], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of strings")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of strings") from exc
    for index, value in enumerate(result):
        _nonempty(value, f"{field_name}[{index}]")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class EditOperation:
    """One forward edit, with cursors positioned before the operation."""

    kind: EditKind
    reference_index: int
    hypothesis_index: int
    reference_character: str | None
    hypothesis_character: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EditKind):
            raise TypeError("kind must be EditKind")
        _plain_int(self.reference_index, "reference_index")
        _plain_int(self.hypothesis_index, "hypothesis_index")
        reference = self.reference_character
        hypothesis = self.hypothesis_character
        if reference is not None and (
            type(reference) is not str or len(reference) != 1
        ):
            raise TypeError("reference_character must be one character or None")
        if hypothesis is not None and (
            type(hypothesis) is not str or len(hypothesis) != 1
        ):
            raise TypeError("hypothesis_character must be one character or None")
        if self.kind in (EditKind.MATCH, EditKind.SUBSTITUTION):
            if reference is None or hypothesis is None:
                raise ValueError("paired edits require both characters")
            if self.kind is EditKind.MATCH and reference != hypothesis:
                raise ValueError("match characters must be equal")
            if self.kind is EditKind.SUBSTITUTION and reference == hypothesis:
                raise ValueError("substitution characters must differ")
        elif self.kind is EditKind.DELETION:
            if reference is None or hypothesis is not None:
                raise ValueError("deletions require only a reference character")
        elif reference is not None or hypothesis is None:
            raise ValueError("insertions require only a hypothesis character")


@dataclass(frozen=True, slots=True)
class EditTrace:
    substitutions: int
    deletions: int
    insertions: int
    operations: tuple[EditOperation, ...]

    def __post_init__(self) -> None:
        for name in ("substitutions", "deletions", "insertions"):
            _plain_int(getattr(self, name), name)
        operations = tuple(self.operations)
        if any(not isinstance(operation, EditOperation) for operation in operations):
            raise TypeError("operations must contain EditOperation values")
        object.__setattr__(self, "operations", operations)
        actual = (
            sum(operation.kind is EditKind.SUBSTITUTION for operation in operations),
            sum(operation.kind is EditKind.DELETION for operation in operations),
            sum(operation.kind is EditKind.INSERTION for operation in operations),
        )
        if actual != (self.substitutions, self.deletions, self.insertions):
            raise ValueError("edit counts do not match operations")

    @property
    def cost(self) -> int:
        return self.substitutions + self.deletions + self.insertions


_CHARACTER_TIE_BREAK = (
    EditKind.MATCH.value,
    EditKind.SUBSTITUTION.value,
    EditKind.DELETION.value,
    EditKind.INSERTION.value,
)
_TRANSITION_KIND_TIE_BREAK = (
    "matched_group",
    "unmatched_reference_event",
    "unmatched_hypothesis",
    "no_score_hypothesis",
)
_GROUP_OBJECTIVE = (
    "character_edit_cost",
    "merge_fragment_excess_members",
    "integer_doubled_center_distance",
    "transition_kind_then_stable_event_hypothesis_ids",
)
_ALIGNMENT_SEMANTICS = (
    "unicode_code_points",
    "collar_is_total_allowed_gap",
    "matched_group_induced_graph_connected",
    "group_limits_not_component_limits",
    "single_pair_group_ignores_horizon",
    "group_horizon_bounds_extent_beyond_longest_member_inclusive",
    "no_score_events_excluded_from_lexical_alignment",
    "hypothesis_with_direct_no_score_edge_may_be_absorbed_at_zero_cost",
    "insertion_to_closest_direct_event_then_earlier_then_id",
    "insertion_inside_one_event_follows_lexical_owner",
)


def _alignment_policy_sha256(
    version: str,
    collar_samples: int,
    max_reference_group: int,
    max_hypothesis_group: int,
    horizon_samples: int,
) -> str:
    return canonical_sha256(
        {
            "character_tie_break": _CHARACTER_TIE_BREAK,
            "collar_samples": collar_samples,
            "group_objective": _GROUP_OBJECTIVE,
            "horizon_samples": horizon_samples,
            "max_hypothesis_group": max_hypothesis_group,
            "max_reference_group": max_reference_group,
            "semantics": _ALIGNMENT_SEMANTICS,
            "transition_kind_tie_break": _TRANSITION_KIND_TIE_BREAK,
            "version": version,
        }
    )


@dataclass(frozen=True, slots=True)
class AlignmentPolicy:
    version: str
    collar_samples: int
    max_reference_group: int
    max_hypothesis_group: int
    horizon_samples: int
    sha256: str

    def __post_init__(self) -> None:
        version = _nonempty(self.version, "version")
        collar = _plain_int(self.collar_samples, "collar_samples")
        reference_group = _plain_int(
            self.max_reference_group, "max_reference_group", minimum=1
        )
        hypothesis_group = _plain_int(
            self.max_hypothesis_group, "max_hypothesis_group", minimum=1
        )
        horizon = _plain_int(self.horizon_samples, "horizon_samples", minimum=1)
        require_sha256(self.sha256, "sha256")
        expected = _alignment_policy_sha256(
            version,
            collar,
            reference_group,
            hypothesis_group,
            horizon,
        )
        if self.sha256 != expected:
            raise ValueError("sha256 does not match the alignment policy")


def _make_alignment_policy(
    version: str,
    collar_samples: int,
    max_reference_group: int,
    max_hypothesis_group: int,
    horizon_samples: int,
) -> AlignmentPolicy:
    return AlignmentPolicy(
        version=version,
        collar_samples=collar_samples,
        max_reference_group=max_reference_group,
        max_hypothesis_group=max_hypothesis_group,
        horizon_samples=horizon_samples,
        sha256=_alignment_policy_sha256(
            version,
            collar_samples,
            max_reference_group,
            max_hypothesis_group,
            horizon_samples,
        ),
    )


JA_TIME_DP_1_0 = _make_alignment_policy(
    "ja-time-dp-1.0",
    collar_samples=4_000,
    max_reference_group=8,
    max_hypothesis_group=8,
    horizon_samples=160_000,
)


@dataclass(frozen=True, slots=True)
class ScoredEvent:
    """One reference event as the aligner sees it.

    ``text`` is the event's own verbatim reference. ``no_score`` marks an event
    with no lexical reference (unintelligible, partially intelligible, or
    coverage-only speech): its span is a no-score region that is excluded from
    lexical alignment. ``text=None`` without ``no_score`` means the event shares
    one verbatim unit with other events, so its lexical ownership is unresolved.
    """

    event_id: str
    span: SampleSpan
    text: str | None
    no_score: bool = False

    def __post_init__(self) -> None:
        require_uuid4(self.event_id, "event_id")
        if not isinstance(self.span, SampleSpan):
            raise TypeError("span must be SampleSpan")
        if self.text is not None and type(self.text) is not str:
            raise TypeError("text must be a string or None")
        if type(self.no_score) is not bool:
            raise TypeError("no_score must be bool")
        if self.no_score and self.text is not None:
            raise ValueError("no-score events carry no lexical reference text")


def _freeze_spans(values: Iterable[object], field_name: str) -> tuple[SampleSpan, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of SampleSpan")
    try:
        spans = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of SampleSpan") from exc
    if any(not isinstance(span, SampleSpan) for span in spans):
        raise TypeError(f"{field_name} must contain SampleSpan values")
    ordered = tuple(
        sorted(spans, key=lambda span: (span.timeline_id, span.start, span.end))
    )
    if ordered:
        timeline = ordered[0].timeline_id
        if any(span.timeline_id != timeline for span in ordered):
            raise ValueError(f"{field_name} must use one timeline")
        if any(
            previous.end > current.start
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise ValueError(f"{field_name} must be ordered non-overlapping spans")
    return ordered


@dataclass(frozen=True, slots=True)
class ScoredHypothesis:
    hypothesis_id: str
    source_spans: tuple[SampleSpan, ...]
    text: str
    synthetic_anchor_timeline_id: str | None = None
    synthetic_anchor_sample: int | None = None

    def __post_init__(self) -> None:
        require_uuid4(self.hypothesis_id, "hypothesis_id")
        spans = _freeze_spans(self.source_spans, "source_spans")
        object.__setattr__(self, "source_spans", spans)
        if type(self.text) is not str:
            raise TypeError("text must be a string")
        if spans:
            if (
                self.synthetic_anchor_timeline_id is not None
                or self.synthetic_anchor_sample is not None
            ):
                raise ValueError("real hypotheses cannot carry a synthetic anchor")
        else:
            require_uuid4(
                self.synthetic_anchor_timeline_id,
                "synthetic_anchor_timeline_id",
            )
            _plain_int(self.synthetic_anchor_sample, "synthetic_anchor_sample")

    @property
    def timeline_id(self) -> str:
        if self.source_spans:
            return self.source_spans[0].timeline_id
        assert self.synthetic_anchor_timeline_id is not None
        return self.synthetic_anchor_timeline_id

    @property
    def extent_start(self) -> int:
        if self.source_spans:
            return self.source_spans[0].start
        assert self.synthetic_anchor_sample is not None
        return self.synthetic_anchor_sample

    @property
    def extent_end(self) -> int:
        if self.source_spans:
            return self.source_spans[-1].end
        assert self.synthetic_anchor_sample is not None
        return self.synthetic_anchor_sample


@dataclass(frozen=True, slots=True)
class DirectEdge:
    event_id: str
    hypothesis_id: str

    def __post_init__(self) -> None:
        require_uuid4(self.event_id, "event_id")
        require_uuid4(self.hypothesis_id, "hypothesis_id")


@dataclass(frozen=True, slots=True)
class TemporalComponent:
    events: tuple[ScoredEvent, ...]
    hypotheses: tuple[ScoredHypothesis, ...]
    direct_edges: tuple[DirectEdge, ...]
    alignment_policy_sha256: str

    def __post_init__(self) -> None:
        events = tuple(self.events)
        hypotheses = tuple(self.hypotheses)
        edges = tuple(self.direct_edges)
        require_sha256(self.alignment_policy_sha256, "alignment_policy_sha256")
        if not events and not hypotheses:
            raise ValueError("a temporal component must not be empty")
        if any(not isinstance(event, ScoredEvent) for event in events):
            raise TypeError("events must contain ScoredEvent values")
        if any(
            not isinstance(hypothesis, ScoredHypothesis) for hypothesis in hypotheses
        ):
            raise TypeError("hypotheses must contain ScoredHypothesis values")
        if any(not isinstance(edge, DirectEdge) for edge in edges):
            raise TypeError("direct_edges must contain DirectEdge values")
        events = tuple(
            sorted(
                events,
                key=lambda event: (
                    event.span.timeline_id,
                    event.span.start,
                    event.span.end,
                    event.event_id,
                ),
            )
        )
        hypotheses = tuple(
            sorted(
                hypotheses,
                key=lambda hypothesis: (
                    hypothesis.timeline_id,
                    hypothesis.extent_start,
                    hypothesis.extent_end,
                    hypothesis.hypothesis_id,
                ),
            )
        )
        event_ids = {event.event_id for event in events}
        hypothesis_ids = {hypothesis.hypothesis_id for hypothesis in hypotheses}
        if len(event_ids) != len(events) or len(hypothesis_ids) != len(hypotheses):
            raise ValueError("component IDs must be unique")
        if event_ids & hypothesis_ids:
            raise ValueError("event and hypothesis IDs must be globally distinct")
        if any(
            edge.event_id not in event_ids or edge.hypothesis_id not in hypothesis_ids
            for edge in edges
        ):
            raise ValueError("direct edge names an item outside the component")
        if len({(edge.event_id, edge.hypothesis_id) for edge in edges}) != len(edges):
            raise ValueError("direct edges must be unique")
        event_order = {event.event_id: index for index, event in enumerate(events)}
        hypothesis_order = {
            hypothesis.hypothesis_id: index
            for index, hypothesis in enumerate(hypotheses)
        }
        edges = tuple(
            sorted(
                edges,
                key=lambda edge: (
                    event_order[edge.event_id],
                    hypothesis_order[edge.hypothesis_id],
                ),
            )
        )
        timelines = {event.span.timeline_id for event in events}
        timelines.update(hypothesis.timeline_id for hypothesis in hypotheses)
        if len(timelines) != 1:
            raise ValueError("a temporal component must use one timeline")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "hypotheses", hypotheses)
        object.__setattr__(self, "direct_edges", edges)


@dataclass(frozen=True, slots=True)
class GroupAlignment:
    event_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    trace: EditTrace
    extent_start: int
    extent_end: int

    def __post_init__(self) -> None:
        event_ids = _freeze_strings(self.event_ids, "event_ids")
        hypothesis_ids = _freeze_strings(self.hypothesis_ids, "hypothesis_ids")
        if not event_ids or not hypothesis_ids:
            raise ValueError("matched groups require events and hypotheses")
        for event_id in event_ids:
            require_uuid4(event_id, "event_ids item")
        for hypothesis_id in hypothesis_ids:
            require_uuid4(hypothesis_id, "hypothesis_ids item")
        if not isinstance(self.trace, EditTrace):
            raise TypeError("trace must be EditTrace")
        start = _plain_int(self.extent_start, "extent_start")
        end = _plain_int(self.extent_end, "extent_end")
        if end < start:
            raise ValueError("group extent must not be reversed")
        object.__setattr__(self, "event_ids", event_ids)
        object.__setattr__(self, "hypothesis_ids", hypothesis_ids)


@dataclass(frozen=True, slots=True)
class EventAlignment:
    event_id: str
    hypothesis_ids: tuple[str, ...]
    substitutions: int
    deletions: int
    insertions: int
    operations: tuple[EditOperation, ...]

    def __post_init__(self) -> None:
        require_uuid4(self.event_id, "event_id")
        hypothesis_ids = _freeze_strings(self.hypothesis_ids, "hypothesis_ids")
        for hypothesis_id in hypothesis_ids:
            require_uuid4(hypothesis_id, "hypothesis_ids item")
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("hypothesis_ids must be unique")
        object.__setattr__(self, "hypothesis_ids", hypothesis_ids)
        trace = EditTrace(
            substitutions=self.substitutions,
            deletions=self.deletions,
            insertions=self.insertions,
            operations=tuple(self.operations),
        )
        object.__setattr__(self, "operations", trace.operations)


@dataclass(frozen=True, slots=True)
class ComponentAlignment:
    """Resolved or unresolved lexical attribution for one temporal component.

    A resolved alignment partitions the component's events into matched groups,
    unmatched events, and no-score events, and its hypotheses into matched
    groups, unmatched hypotheses (insertions), and no-score hypotheses. A
    no-score hypothesis is absorbed by a directly connected no-score event: it
    is neither scored nor counted as an insertion. ``event_alignments`` covers
    exactly the lexically scored events.
    """

    status: AlignmentStatus
    component_event_ids: tuple[str, ...]
    component_hypothesis_ids: tuple[str, ...]
    event_alignments: tuple[EventAlignment, ...]
    groups: tuple[GroupAlignment, ...]
    unmatched_event_ids: tuple[str, ...]
    unmatched_hypothesis_ids: tuple[str, ...]
    substitutions: int | None
    deletions: int | None
    insertions: int | None
    unresolved_reason: str | None = None
    no_score_event_ids: tuple[str, ...] = ()
    no_score_hypothesis_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, AlignmentStatus):
            raise TypeError("status must be AlignmentStatus")
        for field_name in ("component_event_ids", "component_hypothesis_ids"):
            values = _freeze_strings(getattr(self, field_name), field_name)
            for value in values:
                require_uuid4(value, f"{field_name} item")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must contain unique IDs")
            object.__setattr__(self, field_name, values)
        if set(self.component_event_ids) & set(self.component_hypothesis_ids):
            raise ValueError("component event and hypothesis IDs must be disjoint")
        event_alignments = tuple(self.event_alignments)
        groups = tuple(self.groups)
        if any(not isinstance(item, EventAlignment) for item in event_alignments):
            raise TypeError("event_alignments must contain EventAlignment values")
        if any(not isinstance(item, GroupAlignment) for item in groups):
            raise TypeError("groups must contain GroupAlignment values")
        object.__setattr__(self, "event_alignments", event_alignments)
        object.__setattr__(self, "groups", groups)
        for field_name in (
            "unmatched_event_ids",
            "unmatched_hypothesis_ids",
            "no_score_event_ids",
            "no_score_hypothesis_ids",
        ):
            values = _freeze_strings(getattr(self, field_name), field_name)
            for value in values:
                require_uuid4(value, f"{field_name} item")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must contain unique IDs")
            object.__setattr__(self, field_name, values)
        counts = (self.substitutions, self.deletions, self.insertions)
        if self.status is AlignmentStatus.RESOLVED:
            if any(type(value) is not int or value < 0 for value in counts):
                raise ValueError(
                    "resolved alignment counts must be nonnegative integers"
                )
            if self.unresolved_reason is not None:
                raise ValueError("resolved alignments cannot have unresolved_reason")
            no_score_events = set(self.no_score_event_ids)
            if not no_score_events <= set(self.component_event_ids):
                raise ValueError("no-score events must belong to the component")
            aligned_event_ids = tuple(item.event_id for item in event_alignments)
            if (
                len(aligned_event_ids) != len(set(aligned_event_ids))
                or set(aligned_event_ids)
                != set(self.component_event_ids) - no_score_events
            ):
                raise ValueError(
                    "resolved event_alignments must cover each scored event once"
                )
            if any(
                hypothesis_id not in self.component_hypothesis_ids
                for item in event_alignments
                for hypothesis_id in item.hypothesis_ids
            ):
                raise ValueError(
                    "event_alignments cannot name hypotheses outside the component"
                )
            grouped_event_ids = tuple(
                event_id for group in groups for event_id in group.event_ids
            )
            grouped_hypothesis_ids = tuple(
                hypothesis_id
                for group in groups
                for hypothesis_id in group.hypothesis_ids
            )
            if (
                len(grouped_event_ids) != len(set(grouped_event_ids))
                or set(grouped_event_ids) & set(self.unmatched_event_ids)
                or set(grouped_event_ids) | set(self.unmatched_event_ids)
                != set(self.component_event_ids) - no_score_events
            ):
                raise ValueError(
                    "resolved groups, unmatched events, and no-score events must "
                    "partition the component"
                )
            hypothesis_classes = (
                grouped_hypothesis_ids,
                self.unmatched_hypothesis_ids,
                self.no_score_hypothesis_ids,
            )
            classified_hypothesis_ids = tuple(
                hypothesis_id for values in hypothesis_classes for hypothesis_id in values
            )
            if len(classified_hypothesis_ids) != len(
                set(classified_hypothesis_ids)
            ) or set(classified_hypothesis_ids) != set(self.component_hypothesis_ids):
                raise ValueError(
                    "resolved groups, unmatched hypotheses, and no-score hypotheses "
                    "must partition the component"
                )
            group_hypotheses_by_event = {
                event_id: set(group.hypothesis_ids)
                for group in groups
                for event_id in group.event_ids
            }
            for item in event_alignments:
                allowed = group_hypotheses_by_event.get(item.event_id)
                if allowed is None:
                    if item.hypothesis_ids:
                        raise ValueError(
                            "unmatched event alignments cannot name hypotheses"
                        )
                elif not item.hypothesis_ids or not set(item.hypothesis_ids) <= allowed:
                    raise ValueError(
                        "matched event hypotheses must be a nonempty subset of its group"
                    )
        else:
            if counts != (None, None, None):
                raise ValueError("unresolved alignment counts must be None")
            _nonempty(self.unresolved_reason, "unresolved_reason")
            if (
                event_alignments
                or groups
                or self.unmatched_event_ids
                or self.unmatched_hypothesis_ids
                or self.no_score_event_ids
                or self.no_score_hypothesis_ids
            ):
                raise ValueError(
                    "unresolved alignments retain membership without scored attribution"
                )


def character_trace(reference: str, hypothesis: str) -> EditTrace:
    """Return one deterministic minimum edit trace from reference to hypothesis."""

    if type(reference) is not str or type(hypothesis) is not str:
        raise TypeError("reference and hypothesis must be strings")
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    for reference_index in range(len(reference), -1, -1):
        for hypothesis_index in range(len(hypothesis), -1, -1):
            if reference_index == len(reference):
                costs[reference_index][hypothesis_index] = (
                    len(hypothesis) - hypothesis_index
                )
            elif hypothesis_index == len(hypothesis):
                costs[reference_index][hypothesis_index] = (
                    len(reference) - reference_index
                )
            elif reference[reference_index] == hypothesis[hypothesis_index]:
                costs[reference_index][hypothesis_index] = costs[reference_index + 1][
                    hypothesis_index + 1
                ]
            else:
                costs[reference_index][hypothesis_index] = 1 + min(
                    costs[reference_index + 1][hypothesis_index + 1],
                    costs[reference_index + 1][hypothesis_index],
                    costs[reference_index][hypothesis_index + 1],
                )

    operations_list: list[EditOperation] = []
    reference_index = 0
    hypothesis_index = 0
    while reference_index < len(reference) or hypothesis_index < len(hypothesis):
        current = costs[reference_index][hypothesis_index]
        if (
            reference_index < len(reference)
            and hypothesis_index < len(hypothesis)
            and reference[reference_index] == hypothesis[hypothesis_index]
            and current == costs[reference_index + 1][hypothesis_index + 1]
        ):
            kind = EditKind.MATCH
        elif (
            reference_index < len(reference)
            and hypothesis_index < len(hypothesis)
            and reference[reference_index] != hypothesis[hypothesis_index]
            and current == 1 + costs[reference_index + 1][hypothesis_index + 1]
        ):
            kind = EditKind.SUBSTITUTION
        elif (
            reference_index < len(reference)
            and current == 1 + costs[reference_index + 1][hypothesis_index]
        ):
            kind = EditKind.DELETION
        elif (
            hypothesis_index < len(hypothesis)
            and current == 1 + costs[reference_index][hypothesis_index + 1]
        ):
            kind = EditKind.INSERTION
        else:
            raise RuntimeError("edit traceback is incomplete")

        reference_character = (
            reference[reference_index] if kind is not EditKind.INSERTION else None
        )
        hypothesis_character = (
            hypothesis[hypothesis_index] if kind is not EditKind.DELETION else None
        )
        operations_list.append(
            EditOperation(
                kind=kind,
                reference_index=reference_index,
                hypothesis_index=hypothesis_index,
                reference_character=reference_character,
                hypothesis_character=hypothesis_character,
            )
        )
        if kind is not EditKind.INSERTION:
            reference_index += 1
        if kind is not EditKind.DELETION:
            hypothesis_index += 1
    operations = tuple(operations_list)

    reference_cursor = 0
    hypothesis_cursor = 0
    reconstructed_reference: list[str] = []
    reconstructed_hypothesis: list[str] = []
    for operation in operations:
        if (
            operation.reference_index != reference_cursor
            or operation.hypothesis_index != hypothesis_cursor
        ):
            raise RuntimeError("edit traceback cursors are inconsistent")
        if operation.reference_character is not None:
            reconstructed_reference.append(operation.reference_character)
            reference_cursor += 1
        if operation.hypothesis_character is not None:
            reconstructed_hypothesis.append(operation.hypothesis_character)
            hypothesis_cursor += 1
    if (
        "".join(reconstructed_reference) != reference
        or "".join(reconstructed_hypothesis) != hypothesis
    ):
        raise RuntimeError("edit traceback does not reconstruct both strings")
    return EditTrace(
        substitutions=sum(
            operation.kind is EditKind.SUBSTITUTION for operation in operations
        ),
        deletions=sum(operation.kind is EditKind.DELETION for operation in operations),
        insertions=sum(
            operation.kind is EditKind.INSERTION for operation in operations
        ),
        operations=operations,
    )


def _event_key(event: ScoredEvent) -> tuple[str, int, int, str]:
    return (event.span.timeline_id, event.span.start, event.span.end, event.event_id)


def _hypothesis_key(
    hypothesis: ScoredHypothesis,
) -> tuple[str, int, int, str]:
    return (
        hypothesis.timeline_id,
        hypothesis.extent_start,
        hypothesis.extent_end,
        hypothesis.hypothesis_id,
    )


def _span_gap(left: SampleSpan, right: SampleSpan) -> int:
    if left.end <= right.start:
        return right.start - left.end
    if right.end <= left.start:
        return left.start - right.end
    return 0


def _has_direct_edge(
    event: ScoredEvent,
    hypothesis: ScoredHypothesis,
    collar_samples: int,
) -> bool:
    if not hypothesis.source_spans:
        return False
    return any(
        event.span.timeline_id == span.timeline_id
        and _span_gap(event.span, span) <= collar_samples
        for span in hypothesis.source_spans
    )


def build_temporal_components(
    events: Sequence[ScoredEvent],
    hypotheses: Sequence[ScoredHypothesis],
    policy: AlignmentPolicy,
) -> tuple[TemporalComponent, ...]:
    """Build maximal bipartite components from collar-constrained direct edges."""

    if isinstance(events, (str, bytes)) or isinstance(hypotheses, (str, bytes)):
        raise TypeError("events and hypotheses must be sequences")
    if not isinstance(policy, AlignmentPolicy):
        raise TypeError("policy must be AlignmentPolicy")
    event_values = tuple(events)
    hypothesis_values = tuple(hypotheses)
    if any(not isinstance(event, ScoredEvent) for event in event_values):
        raise TypeError("events must contain ScoredEvent values")
    if any(
        not isinstance(hypothesis, ScoredHypothesis) for hypothesis in hypothesis_values
    ):
        raise TypeError("hypotheses must contain ScoredHypothesis values")
    event_ids = [event.event_id for event in event_values]
    hypothesis_ids = [hypothesis.hypothesis_id for hypothesis in hypothesis_values]
    if len(event_ids) != len(set(event_ids)) or len(hypothesis_ids) != len(
        set(hypothesis_ids)
    ):
        raise ValueError("scored IDs must be unique")
    if set(event_ids) & set(hypothesis_ids):
        raise ValueError("event and hypothesis IDs must be globally distinct")

    ordered_events = tuple(sorted(event_values, key=_event_key))
    ordered_hypotheses = tuple(sorted(hypothesis_values, key=_hypothesis_key))
    event_by_id = {event.event_id: event for event in ordered_events}
    hypothesis_by_id = {
        hypothesis.hypothesis_id: hypothesis for hypothesis in ordered_hypotheses
    }
    nodes = {("event", event_id) for event_id in event_by_id}
    nodes.update(("hypothesis", hypothesis_id) for hypothesis_id in hypothesis_by_id)
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = {
        node: set() for node in nodes
    }
    all_edges: list[DirectEdge] = []
    for event in ordered_events:
        for hypothesis in ordered_hypotheses:
            if _has_direct_edge(event, hypothesis, policy.collar_samples):
                edge = DirectEdge(event.event_id, hypothesis.hypothesis_id)
                all_edges.append(edge)
                event_node = ("event", event.event_id)
                hypothesis_node = ("hypothesis", hypothesis.hypothesis_id)
                adjacency[event_node].add(hypothesis_node)
                adjacency[hypothesis_node].add(event_node)

    def node_key(node: tuple[str, str]) -> tuple[str, int, int, int, str]:
        kind, identifier = node
        if kind == "event":
            event = event_by_id[identifier]
            return (
                event.span.timeline_id,
                event.span.start,
                event.span.end,
                0,
                identifier,
            )
        hypothesis = hypothesis_by_id[identifier]
        return (
            hypothesis.timeline_id,
            hypothesis.extent_start,
            hypothesis.extent_end,
            1,
            identifier,
        )

    components: list[TemporalComponent] = []
    remaining = set(nodes)
    while remaining:
        root = min(remaining, key=node_key)
        queue = deque((root,))
        members: set[tuple[str, str]] = set()
        while queue:
            node = queue.popleft()
            if node in members:
                continue
            members.add(node)
            queue.extend(sorted(adjacency[node] - members, key=node_key))
        remaining.difference_update(members)
        member_event_ids = {
            identifier for kind, identifier in members if kind == "event"
        }
        member_hypothesis_ids = {
            identifier for kind, identifier in members if kind == "hypothesis"
        }
        component_events = tuple(
            event for event in ordered_events if event.event_id in member_event_ids
        )
        component_hypotheses = tuple(
            hypothesis
            for hypothesis in ordered_hypotheses
            if hypothesis.hypothesis_id in member_hypothesis_ids
        )
        component_edges = tuple(
            edge
            for edge in all_edges
            if edge.event_id in member_event_ids
            and edge.hypothesis_id in member_hypothesis_ids
        )
        components.append(
            TemporalComponent(
                events=component_events,
                hypotheses=component_hypotheses,
                direct_edges=component_edges,
                alignment_policy_sha256=policy.sha256,
            )
        )
    return tuple(
        sorted(
            components,
            key=lambda component: min(
                (
                    *(
                        (
                            event.span.timeline_id,
                            event.span.start,
                            event.span.end,
                            0,
                            event.event_id,
                        )
                        for event in component.events
                    ),
                    *(
                        (
                            hypothesis.timeline_id,
                            hypothesis.extent_start,
                            hypothesis.extent_end,
                            1,
                            hypothesis.hypothesis_id,
                        )
                        for hypothesis in component.hypotheses
                    ),
                )
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class _Transition:
    kind: str
    events: tuple[ScoredEvent, ...]
    hypotheses: tuple[ScoredHypothesis, ...]
    trace: EditTrace
    reference_owners: tuple[str, ...]
    hypothesis_owners: tuple[str, ...]
    extent_start: int
    extent_end: int
    merge_fragment_count: int
    center_distance: int
    stable_key: str


@dataclass(frozen=True, slots=True)
class _Step:
    kind: str
    reference_size: int
    hypothesis_size: int
    edit_cost: int
    merge_fragment_count: int
    center_distance: int
    stable_key: str

    @property
    def objective(self) -> tuple[int, int, int]:
        return (
            self.edit_cost,
            self.merge_fragment_count,
            self.center_distance,
        )


def _extent_of_items(
    events: Sequence[ScoredEvent], hypotheses: Sequence[ScoredHypothesis]
) -> tuple[int, int]:
    starts = [event.span.start for event in events]
    ends = [event.span.end for event in events]
    starts.extend(hypothesis.extent_start for hypothesis in hypotheses)
    ends.extend(hypothesis.extent_end for hypothesis in hypotheses)
    return min(starts), max(ends)


def _within_horizon(
    events: Sequence[ScoredEvent],
    hypotheses: Sequence[ScoredHypothesis],
    policy: AlignmentPolicy,
) -> bool:
    """Bound how far a merge or fragment group reaches beyond its longest member.

    A single event matched to a single hypothesis is always admissible, however
    long either one is. A larger group may extend at most ``horizon_samples``
    beyond the span of its longest member.
    """

    if len(events) == 1 and len(hypotheses) == 1:
        return True
    extent_start, extent_end = _extent_of_items(events, hypotheses)
    longest_member = max(
        [event.span.end - event.span.start for event in events]
        + [item.extent_end - item.extent_start for item in hypotheses]
    )
    return extent_end - extent_start - longest_member <= policy.horizon_samples


def _group_connected(
    events: Sequence[ScoredEvent],
    hypotheses: Sequence[ScoredHypothesis],
    edge_pairs: set[tuple[str, str]],
) -> bool:
    event_ids = {event.event_id for event in events}
    hypothesis_ids = {hypothesis.hypothesis_id for hypothesis in hypotheses}
    if any(
        not any(
            (event_id, hypothesis_id) in edge_pairs for hypothesis_id in hypothesis_ids
        )
        for event_id in event_ids
    ) or any(
        not any((event_id, hypothesis_id) in edge_pairs for event_id in event_ids)
        for hypothesis_id in hypothesis_ids
    ):
        return False
    nodes = {("event", identifier) for identifier in event_ids}
    nodes.update(("hypothesis", identifier) for identifier in hypothesis_ids)
    adjacency = {node: set() for node in nodes}
    for event_id in event_ids:
        for hypothesis_id in hypothesis_ids:
            if (event_id, hypothesis_id) in edge_pairs:
                left = ("event", event_id)
                right = ("hypothesis", hypothesis_id)
                adjacency[left].add(right)
                adjacency[right].add(left)
    root = next(iter(nodes))
    seen: set[tuple[str, str]] = set()
    queue = deque((root,))
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        queue.extend(adjacency[node] - seen)
    return seen == nodes


def _component_connected(
    events: Sequence[ScoredEvent],
    hypotheses: Sequence[ScoredHypothesis],
    edge_pairs: set[tuple[str, str]],
) -> bool:
    if events and hypotheses:
        return _group_connected(events, hypotheses, edge_pairs)
    return len(events) + len(hypotheses) == 1 and not edge_pairs


def _policy_edge_pairs(
    events: Sequence[ScoredEvent],
    hypotheses: Sequence[ScoredHypothesis],
    policy: AlignmentPolicy,
) -> set[tuple[str, str]]:
    return {
        (event.event_id, hypothesis.hypothesis_id)
        for event in events
        for hypothesis in hypotheses
        if _has_direct_edge(event, hypothesis, policy.collar_samples)
    }


def _normalized(text: str, normalizer: Callable[[str], str]) -> str:
    value = normalizer(text)
    if type(value) is not str:
        raise TypeError("normalizer must return a string")
    return value


def _character_distance(reference: str, hypothesis: str) -> int:
    """Return edit cost without retaining the full character traceback."""

    if len(hypothesis) > len(reference):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_character in enumerate(reference, 1):
        current = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_character != hypothesis_character),
                )
            )
        previous = current
    return previous[-1]


def _matched_step(
    events: tuple[ScoredEvent, ...],
    hypotheses: tuple[ScoredHypothesis, ...],
    edge_pairs: set[tuple[str, str]],
    policy: AlignmentPolicy,
    normalized_events: dict[str, str],
    normalized_hypotheses: dict[str, str],
) -> _Step | None:
    if not _group_connected(events, hypotheses, edge_pairs):
        return None
    if not _within_horizon(events, hypotheses, policy):
        return None
    reference = "".join(normalized_events[event.event_id] for event in events)
    hypothesis = "".join(
        normalized_hypotheses[item.hypothesis_id] for item in hypotheses
    )
    event_start = min(event.span.start for event in events)
    event_end = max(event.span.end for event in events)
    hypothesis_start = min(item.extent_start for item in hypotheses)
    hypothesis_end = max(item.extent_end for item in hypotheses)
    stable_key = (
        "0:match:E="
        + ",".join(event.event_id for event in events)
        + ":H="
        + ",".join(item.hypothesis_id for item in hypotheses)
    )
    return _Step(
        kind="match",
        reference_size=len(events),
        hypothesis_size=len(hypotheses),
        edit_cost=_character_distance(reference, hypothesis),
        merge_fragment_count=len(events) + len(hypotheses) - 2,
        center_distance=abs(
            (event_start + event_end) - (hypothesis_start + hypothesis_end)
        ),
        stable_key=stable_key,
    )


def _matched_transition(
    events: tuple[ScoredEvent, ...],
    hypotheses: tuple[ScoredHypothesis, ...],
    edge_pairs: set[tuple[str, str]],
    policy: AlignmentPolicy,
    normalized_events: dict[str, str],
    normalized_hypotheses: dict[str, str],
) -> _Transition | None:
    if not _group_connected(events, hypotheses, edge_pairs):
        return None
    if not _within_horizon(events, hypotheses, policy):
        return None
    extent_start, extent_end = _extent_of_items(events, hypotheses)
    reference_parts = [normalized_events[event.event_id] for event in events]
    hypothesis_parts = [
        normalized_hypotheses[hypothesis.hypothesis_id] for hypothesis in hypotheses
    ]
    reference_owners = tuple(
        event.event_id for event, text in zip(events, reference_parts) for _ in text
    )
    hypothesis_owners = tuple(
        hypothesis.hypothesis_id
        for hypothesis, text in zip(hypotheses, hypothesis_parts)
        for _ in text
    )
    trace = character_trace("".join(reference_parts), "".join(hypothesis_parts))
    event_start = min(event.span.start for event in events)
    event_end = max(event.span.end for event in events)
    hypothesis_start = min(hypothesis.extent_start for hypothesis in hypotheses)
    hypothesis_end = max(hypothesis.extent_end for hypothesis in hypotheses)
    stable_key = (
        "0:match:E="
        + ",".join(event.event_id for event in events)
        + ":H="
        + ",".join(hypothesis.hypothesis_id for hypothesis in hypotheses)
    )
    return _Transition(
        kind="match",
        events=events,
        hypotheses=hypotheses,
        trace=trace,
        reference_owners=reference_owners,
        hypothesis_owners=hypothesis_owners,
        extent_start=extent_start,
        extent_end=extent_end,
        merge_fragment_count=len(events) + len(hypotheses) - 2,
        center_distance=abs(
            (event_start + event_end) - (hypothesis_start + hypothesis_end)
        ),
        stable_key=stable_key,
    )


def _unmatched_event_transition(event: ScoredEvent, normalized: str) -> _Transition:
    trace = character_trace(normalized, "")
    return _Transition(
        kind="event",
        events=(event,),
        hypotheses=(),
        trace=trace,
        reference_owners=tuple(event.event_id for _ in normalized),
        hypothesis_owners=(),
        extent_start=event.span.start,
        extent_end=event.span.end,
        merge_fragment_count=0,
        center_distance=0,
        stable_key=f"1:event:{event.event_id}",
    )


def _unmatched_hypothesis_transition(
    hypothesis: ScoredHypothesis, normalized: str
) -> _Transition:
    trace = character_trace("", normalized)
    return _Transition(
        kind="hypothesis",
        events=(),
        hypotheses=(hypothesis,),
        trace=trace,
        reference_owners=(),
        hypothesis_owners=tuple(hypothesis.hypothesis_id for _ in normalized),
        extent_start=hypothesis.extent_start,
        extent_end=hypothesis.extent_end,
        merge_fragment_count=0,
        center_distance=0,
        stable_key=f"2:hypothesis:{hypothesis.hypothesis_id}",
    )


def _no_score_hypothesis_transition(hypothesis: ScoredHypothesis) -> _Transition:
    return _Transition(
        kind="no_score",
        events=(),
        hypotheses=(hypothesis,),
        trace=character_trace("", ""),
        reference_owners=(),
        hypothesis_owners=(),
        extent_start=hypothesis.extent_start,
        extent_end=hypothesis.extent_end,
        merge_fragment_count=0,
        center_distance=0,
        stable_key=f"3:no_score:{hypothesis.hypothesis_id}",
    )


def _unmatched_event_step(event: ScoredEvent, normalized: str) -> _Step:
    return _Step(
        kind="event",
        reference_size=1,
        hypothesis_size=0,
        edit_cost=len(normalized),
        merge_fragment_count=0,
        center_distance=0,
        stable_key=f"1:event:{event.event_id}",
    )


def _unmatched_hypothesis_step(hypothesis: ScoredHypothesis, normalized: str) -> _Step:
    return _Step(
        kind="hypothesis",
        reference_size=0,
        hypothesis_size=1,
        edit_cost=len(normalized),
        merge_fragment_count=0,
        center_distance=0,
        stable_key=f"2:hypothesis:{hypothesis.hypothesis_id}",
    )


def _no_score_hypothesis_step(hypothesis: ScoredHypothesis) -> _Step:
    return _Step(
        kind="no_score",
        reference_size=0,
        hypothesis_size=1,
        edit_cost=0,
        merge_fragment_count=0,
        center_distance=0,
        stable_key=f"3:no_score:{hypothesis.hypothesis_id}",
    )


def _add_objective(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


@dataclass(slots=True)
class _EventAccumulator:
    hypothesis_ids: tuple[str, ...]
    operations: list[EditOperation]


def _operation_owner(
    operation: EditOperation,
    transition: _Transition,
    edge_pairs: set[tuple[str, str]],
) -> str:
    if operation.kind is not EditKind.INSERTION:
        return transition.reference_owners[operation.reference_index]
    hypothesis_id = transition.hypothesis_owners[operation.hypothesis_index]
    adjacent_owners: set[str] = set()
    if operation.reference_index > 0:
        adjacent_owners.add(
            transition.reference_owners[operation.reference_index - 1]
        )
    if operation.reference_index < len(transition.reference_owners):
        adjacent_owners.add(transition.reference_owners[operation.reference_index])
    if len(adjacent_owners) == 1:
        lexical_owner = next(iter(adjacent_owners))
        if (lexical_owner, hypothesis_id) in edge_pairs:
            return lexical_owner
    hypothesis = next(
        item for item in transition.hypotheses if item.hypothesis_id == hypothesis_id
    )
    candidates = tuple(
        event
        for event in transition.events
        if (event.event_id, hypothesis_id) in edge_pairs
    )
    if not candidates:
        raise RuntimeError("matched insertion has no directly connected event")
    hypothesis_center = hypothesis.extent_start + hypothesis.extent_end
    return min(
        candidates,
        key=lambda event: (
            abs((event.span.start + event.span.end) - hypothesis_center),
            event.span.start,
            event.span.end,
            event.event_id,
        ),
    ).event_id


def _unresolved(component: TemporalComponent, reason: str) -> ComponentAlignment:
    return ComponentAlignment(
        status=AlignmentStatus.ALIGNMENT_UNRESOLVED,
        component_event_ids=tuple(event.event_id for event in component.events),
        component_hypothesis_ids=tuple(
            hypothesis.hypothesis_id for hypothesis in component.hypotheses
        ),
        event_alignments=(),
        groups=(),
        unmatched_event_ids=(),
        unmatched_hypothesis_ids=(),
        substitutions=None,
        deletions=None,
        insertions=None,
        unresolved_reason=reason,
    )


def align_component(
    component: TemporalComponent,
    policy: AlignmentPolicy,
    normalizer: Callable[[str], str],
) -> ComponentAlignment:
    """Align one temporal component without consuming any item more than once."""

    if not isinstance(component, TemporalComponent):
        raise TypeError("component must be TemporalComponent")
    if not isinstance(policy, AlignmentPolicy):
        raise TypeError("policy must be AlignmentPolicy")
    if not callable(normalizer):
        raise TypeError("normalizer must be callable")
    if component.alignment_policy_sha256 != policy.sha256:
        return _unresolved(component, "component_policy_mismatch")
    if any(event.text is None and not event.no_score for event in component.events):
        return _unresolved(component, "event_lexical_ownership_unresolved")

    hypotheses = component.hypotheses
    edge_pairs = _policy_edge_pairs(component.events, hypotheses, policy)
    supplied_edge_pairs = {
        (edge.event_id, edge.hypothesis_id) for edge in component.direct_edges
    }
    if edge_pairs != supplied_edge_pairs or not _component_connected(
        component.events, hypotheses, edge_pairs
    ):
        return _unresolved(component, "component_policy_mismatch")
    # No-score events have no lexical reference. They stay in the component for
    # membership and fates, but only lexically scored events enter the program.
    # A hypothesis directly connected to a no-score event may be absorbed by that
    # region at zero cost, so text spoken there is neither scored nor inserted.
    no_score_event_ids = tuple(
        event.event_id for event in component.events if event.no_score
    )
    absorbable_hypothesis_ids = frozenset(
        hypothesis_id
        for event_id, hypothesis_id in edge_pairs
        if event_id in no_score_event_ids
    )
    events = tuple(event for event in component.events if not event.no_score)
    normalized_events = {
        event.event_id: _normalized(event.text, normalizer)  # type: ignore[arg-type]
        for event in events
    }
    normalized_hypotheses = {
        hypothesis.hypothesis_id: _normalized(hypothesis.text, normalizer)
        for hypothesis in hypotheses
    }
    event_count = len(events)
    hypothesis_count = len(hypotheses)
    objectives: list[list[tuple[int, int, int] | None]] = [
        [None] * (hypothesis_count + 1) for _ in range(event_count + 1)
    ]
    steps: list[list[_Step | None]] = [
        [None] * (hypothesis_count + 1) for _ in range(event_count + 1)
    ]
    objectives[event_count][hypothesis_count] = (0, 0, 0)
    # Group bounds define the legal scoring graph. Unmatched moves keep that
    # graph complete without admitting an unconstrained fallback alignment.
    for event_index in range(event_count, -1, -1):
        for hypothesis_index in range(hypothesis_count, -1, -1):
            if event_index == event_count and hypothesis_index == hypothesis_count:
                continue
            candidates: list[tuple[tuple[int, int, int], str, _Step]] = []
            if event_index < event_count:
                child = objectives[event_index + 1][hypothesis_index]
                if child is not None:
                    event = events[event_index]
                    step = _unmatched_event_step(
                        event, normalized_events[event.event_id]
                    )
                    candidates.append(
                        (_add_objective(step.objective, child), step.stable_key, step)
                    )
            if hypothesis_index < hypothesis_count:
                child = objectives[event_index][hypothesis_index + 1]
                if child is not None:
                    hypothesis = hypotheses[hypothesis_index]
                    step = _unmatched_hypothesis_step(
                        hypothesis,
                        normalized_hypotheses[hypothesis.hypothesis_id],
                    )
                    candidates.append(
                        (_add_objective(step.objective, child), step.stable_key, step)
                    )
                    if hypothesis.hypothesis_id in absorbable_hypothesis_ids:
                        step = _no_score_hypothesis_step(hypothesis)
                        candidates.append(
                            (
                                _add_objective(step.objective, child),
                                step.stable_key,
                                step,
                            )
                        )
            max_events = min(policy.max_reference_group, event_count - event_index)
            max_hypotheses = min(
                policy.max_hypothesis_group,
                hypothesis_count - hypothesis_index,
            )
            for reference_size in range(1, max_events + 1):
                for hypothesis_size in range(1, max_hypotheses + 1):
                    child = objectives[event_index + reference_size][
                        hypothesis_index + hypothesis_size
                    ]
                    if child is None:
                        continue
                    step = _matched_step(
                        events[event_index : event_index + reference_size],
                        hypotheses[
                            hypothesis_index : hypothesis_index + hypothesis_size
                        ],
                        edge_pairs,
                        policy,
                        normalized_events,
                        normalized_hypotheses,
                    )
                    if step is not None:
                        candidates.append(
                            (
                                _add_objective(step.objective, child),
                                step.stable_key,
                                step,
                            )
                        )
            if not candidates:
                raise RuntimeError("bounded alignment DP has no legal transition")
            objective, _, step = min(
                candidates, key=lambda candidate: (candidate[0], candidate[1])
            )
            objectives[event_index][hypothesis_index] = objective
            steps[event_index][hypothesis_index] = step

    selected_objective = objectives[0][0]
    if selected_objective is None:
        raise RuntimeError("bounded alignment DP did not reach the initial state")
    selected_transitions: list[_Transition] = []
    event_index = 0
    hypothesis_index = 0
    while event_index < event_count or hypothesis_index < hypothesis_count:
        step = steps[event_index][hypothesis_index]
        if step is None:
            raise RuntimeError("alignment backpointer is incomplete")
        if step.kind == "match":
            transition = _matched_transition(
                events[event_index : event_index + step.reference_size],
                hypotheses[hypothesis_index : hypothesis_index + step.hypothesis_size],
                edge_pairs,
                policy,
                normalized_events,
                normalized_hypotheses,
            )
            if transition is None:
                raise RuntimeError("selected matched group no longer satisfies bounds")
        elif step.kind == "event":
            event = events[event_index]
            transition = _unmatched_event_transition(
                event, normalized_events[event.event_id]
            )
        elif step.kind == "hypothesis":
            hypothesis = hypotheses[hypothesis_index]
            transition = _unmatched_hypothesis_transition(
                hypothesis,
                normalized_hypotheses[hypothesis.hypothesis_id],
            )
        elif step.kind == "no_score":
            transition = _no_score_hypothesis_transition(hypotheses[hypothesis_index])
        else:
            raise RuntimeError("alignment backpointer has an unknown step")
        if (
            transition.trace.cost != step.edit_cost
            or transition.merge_fragment_count != step.merge_fragment_count
            or transition.center_distance != step.center_distance
            or transition.stable_key != step.stable_key
        ):
            raise RuntimeError("materialized transition differs from its DP summary")
        selected_transitions.append(transition)
        event_index += step.reference_size
        hypothesis_index += step.hypothesis_size

    consumed_events = tuple(
        event.event_id
        for transition in selected_transitions
        for event in transition.events
    )
    consumed_hypotheses = tuple(
        hypothesis.hypothesis_id
        for transition in selected_transitions
        for hypothesis in transition.hypotheses
    )
    if (
        sorted(consumed_events) != sorted(event.event_id for event in events)
        or sorted(consumed_hypotheses)
        != sorted(hypothesis.hypothesis_id for hypothesis in hypotheses)
        or len(consumed_events) != len(set(consumed_events))
        or len(consumed_hypotheses) != len(set(consumed_hypotheses))
    ):
        raise RuntimeError("alignment transitions do not partition the component")

    accumulators: dict[str, _EventAccumulator] = {}
    groups: list[GroupAlignment] = []
    unmatched_event_ids: list[str] = []
    unmatched_hypothesis_ids: list[str] = []
    no_score_hypothesis_ids: list[str] = []
    for transition in selected_transitions:
        if transition.kind == "match":
            groups.append(
                GroupAlignment(
                    event_ids=tuple(event.event_id for event in transition.events),
                    hypothesis_ids=tuple(
                        hypothesis.hypothesis_id for hypothesis in transition.hypotheses
                    ),
                    trace=transition.trace,
                    extent_start=transition.extent_start,
                    extent_end=transition.extent_end,
                )
            )
            for event in transition.events:
                direct_hypotheses = tuple(
                    hypothesis.hypothesis_id
                    for hypothesis in transition.hypotheses
                    if (event.event_id, hypothesis.hypothesis_id) in edge_pairs
                )
                accumulators[event.event_id] = _EventAccumulator(direct_hypotheses, [])
            for operation in transition.trace.operations:
                owner = _operation_owner(operation, transition, edge_pairs)
                accumulators[owner].operations.append(operation)
        elif transition.kind == "event":
            event = transition.events[0]
            unmatched_event_ids.append(event.event_id)
            accumulators[event.event_id] = _EventAccumulator(
                (), list(transition.trace.operations)
            )
        elif transition.kind == "hypothesis":
            unmatched_hypothesis_ids.append(transition.hypotheses[0].hypothesis_id)
        elif transition.kind == "no_score":
            no_score_hypothesis_ids.append(transition.hypotheses[0].hypothesis_id)
        else:
            raise RuntimeError("unknown alignment transition")

    event_alignments: list[EventAlignment] = []
    for event in events:
        accumulator = accumulators.get(event.event_id)
        if accumulator is None:
            raise RuntimeError("alignment omitted an event attribution")
        operations = tuple(accumulator.operations)
        event_alignments.append(
            EventAlignment(
                event_id=event.event_id,
                hypothesis_ids=accumulator.hypothesis_ids,
                substitutions=sum(
                    operation.kind is EditKind.SUBSTITUTION for operation in operations
                ),
                deletions=sum(
                    operation.kind is EditKind.DELETION for operation in operations
                ),
                insertions=sum(
                    operation.kind is EditKind.INSERTION for operation in operations
                ),
                operations=operations,
            )
        )

    substitutions = sum(
        transition.trace.substitutions for transition in selected_transitions
    )
    deletions = sum(transition.trace.deletions for transition in selected_transitions)
    insertions = sum(transition.trace.insertions for transition in selected_transitions)
    if substitutions + deletions + insertions != selected_objective[0]:
        raise RuntimeError("alignment totals do not match the selected objective")
    if sum(item.substitutions for item in event_alignments) != substitutions:
        raise RuntimeError("event substitution attribution does not reconcile")
    if sum(item.deletions for item in event_alignments) != deletions:
        raise RuntimeError("event deletion attribution does not reconcile")
    attributed_insertions = sum(item.insertions for item in event_alignments)
    orphan_insertions = sum(
        transition.trace.insertions
        for transition in selected_transitions
        if transition.kind == "hypothesis"
    )
    if attributed_insertions + orphan_insertions != insertions:
        raise RuntimeError("event insertion attribution does not reconcile")
    return ComponentAlignment(
        status=AlignmentStatus.RESOLVED,
        component_event_ids=tuple(event.event_id for event in component.events),
        component_hypothesis_ids=tuple(
            hypothesis.hypothesis_id for hypothesis in hypotheses
        ),
        event_alignments=tuple(event_alignments),
        groups=tuple(groups),
        unmatched_event_ids=tuple(unmatched_event_ids),
        unmatched_hypothesis_ids=tuple(unmatched_hypothesis_ids),
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        no_score_event_ids=no_score_event_ids,
        no_score_hypothesis_ids=tuple(no_score_hypothesis_ids),
    )


__all__ = (
    "AlignmentPolicy",
    "AlignmentStatus",
    "ComponentAlignment",
    "DirectEdge",
    "EditKind",
    "EditOperation",
    "EditTrace",
    "EventAlignment",
    "GroupAlignment",
    "JA_TIME_DP_1_0",
    "ScoredEvent",
    "ScoredHypothesis",
    "TemporalComponent",
    "align_component",
    "build_temporal_components",
    "character_trace",
)
