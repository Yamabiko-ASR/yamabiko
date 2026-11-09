"""Score verified E0 runs from committed ledger, registry, and reference data.

Every ``ScoredFixture`` input is rebuilt from what a verified run and the
private registry already hold. This module must not import ``runtime``, which
constructs the production VAD and ASR.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from scripts.e0_evidence.align import (
    JA_TIME_DP_1_0,
    AlignmentStatus,
    ComponentAlignment,
    EventAlignment,
    ScoredEvent,
    ScoredHypothesis,
    align_component,
    build_temporal_components,
)
from scripts.e0_evidence.ledger import iter_canonical_records, iter_private_text_records
from scripts.e0_evidence.normalize_ja import (
    JA_LENIENT_1_1,
    JA_MINIMAL_1_1,
    JA_WORDS_1_0,
    normalize_lenient,
    normalize_minimal,
    normalize_raw,
    normalize_words,
    split_vocal_sounds,
)
from scripts.e0_evidence.pcm import load_canonical_master
from scripts.e0_evidence.private_root import PrivateBoundary
from scripts.e0_evidence.references import ReferenceEvent, load_reference_set
from scripts.e0_evidence.registry import (
    load_reference_texts,
    load_registered_fixture,
    load_registered_master,
    load_registered_reference,
)
from scripts.e0_evidence.report import (
    PrivateReportError,
    VerifiedSource,
    verify_report_sources,
    write_score_bundle,
)
from scripts.e0_evidence.schema import (
    ArmKind,
    ArmStatus,
    DiagnosticCode,
    EventFate,
    EvidenceRecord,
    FixtureManifest,
    ObservationStatus,
    RecordKind,
    SampleSpan,
    TerminalOutcome,
    TextRole,
    new_opaque_id,
)
from scripts.e0_evidence.score import (
    JA_EVENT_USABLE_1_1,
    RAW_UNICODE_1_0_SHA256,
    RAW_UNICODE_1_0_VERSION,
    EventScore,
    NormalizationLevel,
    NormalizedAlignmentSet,
    ResourceObservations,
    ScalarObservation,
    ScoredFixture,
    ScoreReport,
    VocalSoundCounts,
    compute_metrics,
    duplicate_event_ids,
    event_is_eligible,
    event_is_lexically_usable,
    scored_reference_events,
)
from scripts.e0_evidence.wire import canonical_sha256


_HARD_STRATUM_KEYS = frozenset({"version", "tags"})

# E0 replays a master offline and synchronously, not paced to the source clock
# (runbook section 9), so event-end-to-output latency is outside its scope.
_UNPACED_LATENCY = ScalarObservation(ObservationStatus.OUT_OF_SCOPE, None, "seconds")
_REPLAY_ARM_KINDS = frozenset({ArmKind.P0, ArmKind.P2A, ArmKind.P2B})
_NORMALIZATIONS = (
    (
        NormalizationLevel.RAW,
        RAW_UNICODE_1_0_VERSION,
        RAW_UNICODE_1_0_SHA256,
        normalize_raw,
    ),
    (
        NormalizationLevel.MINIMAL,
        JA_MINIMAL_1_1.version,
        JA_MINIMAL_1_1.sha256,
        normalize_minimal,
    ),
    (
        NormalizationLevel.LENIENT,
        JA_LENIENT_1_1.version,
        JA_LENIENT_1_1.sha256,
        normalize_lenient,
    ),
    (
        NormalizationLevel.WORDS,
        JA_WORDS_1_0.version,
        JA_WORDS_1_0.sha256,
        normalize_words,
    ),
)
_RESOURCE_UNITS = (
    ("rtf", "ratio"),
    ("cold_start_seconds", "seconds"),
    ("peak_rss_bytes", "bytes"),
    ("peak_vram_bytes", "bytes"),
)


@dataclass(frozen=True, slots=True)
class RunScores:
    """One report per scoreable arm, keyed for ``write_score_bundle``."""

    reports: tuple[ScoreReport, ...]
    score_sources: Mapping[str, tuple[str, ...]]
    omitted_stochastic_arm_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScoreBundle:
    report_id: str
    bundle_path: Path
    omitted_stochastic_arm_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Fixture:
    """One verified source run joined to its frozen reference and registry."""

    source: VerifiedSource
    manifest: FixtureManifest
    timeline_id: str
    sample_rate_hz: int
    sample_count: int
    events: tuple[ReferenceEvent, ...]
    scored_events: tuple[ScoredEvent, ...]
    non_speech_control_spans: tuple[SampleSpan, ...]
    hard_stratum_version: str
    hard_stratum_tags: tuple[str, ...]
    records: tuple[EvidenceRecord, ...]
    raw_texts: Mapping[str, str]


def _hard_stratum(details: Mapping[str, object]) -> tuple[str, tuple[str, ...]]:
    """Return the hard-stratum version and tags the cohort froze for this run."""

    declared = details.get("hard_stratum")
    if not isinstance(declared, Mapping) or set(declared) != _HARD_STRATUM_KEYS:
        raise PrivateReportError("run start does not declare its hard stratum")
    version = declared["version"]
    tags = declared["tags"]
    if (
        type(version) is not str
        or not version
        or isinstance(tags, (str, bytes))
        or not isinstance(tags, Sequence)
        or not tags
        or any(type(tag) is not str for tag in tags)
        or list(tags) != sorted(set(tags))
    ):
        raise PrivateReportError("run start hard stratum is invalid")
    return version, tuple(tags)


def _load_fixture(source: VerifiedSource, boundary: PrivateBoundary) -> _Fixture:
    details = source.start.details
    master_id = details.get("master_id")
    reference_set_id = details.get("reference_set_id")
    if type(master_id) is not str or type(reference_set_id) is not str:
        raise PrivateReportError("run start does not name its master and reference")
    # The same frozen inputs ``run`` loads before inference; the reference
    # loader enforces the partition's double-annotation rule.
    master_manifest = load_registered_master(master_id, boundary)
    reference_manifest = load_registered_reference(reference_set_id, boundary)
    reference_texts = load_reference_texts(reference_set_id, boundary)
    canonical_master = load_canonical_master(master_manifest, boundary)
    reference_set = load_reference_set(reference_manifest, canonical_master)
    if (
        reference_manifest.reference_set_sha256 != details.get("reference_set_sha256")
        or reference_manifest.fixture_id != source.fixture_id
        or master_manifest.canonical_pcm_sha256 != details.get("master_pcm_sha256")
        or tuple(event.event_id for event in reference_set.events)
        != details.get("reference_event_ids")
    ):
        raise PrivateReportError("frozen reference does not match the run start")
    fixture = load_registered_fixture(source.fixture_id, boundary)
    if fixture.partition is not source.partition:
        raise PrivateReportError("registered fixture partition does not match the run")
    timeline = canonical_master.manifest
    hard_stratum_version, hard_stratum_tags = _hard_stratum(details)
    return _Fixture(
        source=source,
        manifest=fixture,
        timeline_id=timeline.timeline_id,
        sample_rate_hz=timeline.sample_rate_hz,
        sample_count=timeline.sample_count,
        events=reference_set.events,
        scored_events=scored_reference_events(reference_set, reference_texts),
        non_speech_control_spans=tuple(
            control.span for control in reference_set.non_speech_controls
        ),
        hard_stratum_version=hard_stratum_version,
        hard_stratum_tags=hard_stratum_tags,
        records=tuple(iter_canonical_records(source.run_path)),
        raw_texts={
            record.canonical_record_id: record.text
            for record in iter_private_text_records(source.run_path)
            if record.role is TextRole.RAW_HYPOTHESIS
        },
    )


def _observed(
    fixture: _Fixture,
    arm_id: str,
    kind: RecordKind,
) -> tuple[EvidenceRecord, ...]:
    return tuple(
        record
        for record in fixture.records
        if record.arm_id == arm_id
        and record.kind is kind
        and record.observation_status is ObservationStatus.OBSERVED
    )


def _raw_hypotheses(
    fixture: _Fixture, arm_id: str
) -> tuple[tuple[ScoredHypothesis, ...], dict[str, str]]:
    """Return the arm's raw hypotheses and each one's producing decoder call."""

    hypotheses: list[ScoredHypothesis] = []
    decoder_calls: dict[str, str] = {}
    for record in _observed(fixture, arm_id, RecordKind.RAW_HYPOTHESIS):
        text = fixture.raw_texts.get(record.record_id)
        if text is None:
            raise PrivateReportError("a raw hypothesis has no private text")
        if not record.parent_ids:
            raise PrivateReportError("a raw hypothesis names no decoder call")
        decoder_calls[record.record_id] = record.parent_ids[0]
        hypotheses.append(
            ScoredHypothesis(
                hypothesis_id=record.record_id,
                source_spans=record.source_spans,
                text=text,
                synthetic_anchor_timeline_id=cast(
                    str | None,
                    record.details.get("synthetic_anchor_timeline_id"),
                ),
                synthetic_anchor_sample=cast(
                    int | None,
                    record.details.get("synthetic_anchor_sample"),
                ),
            )
        )
    return tuple(hypotheses), decoder_calls


def _wrapper_accepted_ids(
    fixture: _Fixture,
    arm_id: str,
    raw_ids: Iterable[str],
) -> tuple[str, ...]:
    # A terminal available decision names exactly the raw records the wrapper
    # kept, so confidence-gated or compression-rejected text is never credited.
    known = set(raw_ids)
    accepted: list[str] = []
    for record in _observed(fixture, arm_id, RecordKind.WRAPPER_DECISION):
        if record.decision != TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value:
            continue
        for parent_id in record.parent_ids:
            if parent_id not in known or parent_id in accepted:
                raise PrivateReportError(
                    "wrapper-accepted parents are not distinct raw hypotheses"
                )
            accepted.append(parent_id)
    return tuple(accepted)


def _normalized_alignments(
    events: tuple[ScoredEvent, ...],
    hypotheses: tuple[ScoredHypothesis, ...],
) -> tuple[NormalizedAlignmentSet, ...]:
    components = build_temporal_components(events, hypotheses, JA_TIME_DP_1_0)
    return tuple(
        NormalizedAlignmentSet(
            level=level,
            normalization_policy_version=version,
            normalization_policy_sha256=sha256,
            alignment_policy_version=JA_TIME_DP_1_0.version,
            alignment_policy_sha256=JA_TIME_DP_1_0.sha256,
            components=tuple(
                align_component(component, JA_TIME_DP_1_0, normalizer)
                for component in components
            ),
        )
        for level, version, sha256, normalizer in _NORMALIZATIONS
    )


def _event_alignments(
    components: Iterable[ComponentAlignment],
) -> dict[str, EventAlignment | None]:
    """Map lexically scored events to their alignment; unresolved ones to None."""

    result: dict[str, EventAlignment | None] = {}
    for component in components:
        if component.status is AlignmentStatus.RESOLVED:
            result.update((item.event_id, item) for item in component.event_alignments)
        else:
            result.update((event_id, None) for event_id in component.component_event_ids)
    return result


def _event_fates(fixture: _Fixture, arm_id: str) -> dict[str, EventFate]:
    fates: dict[str, EventFate] = {}
    for record in fixture.records:
        if record.arm_id != arm_id or record.kind is not RecordKind.EVENT_FATE:
            continue
        try:
            outcome = TerminalOutcome(record.decision)
        except ValueError:
            raise PrivateReportError("an event fate has no terminal outcome") from None
        if record.artifact_id in fates or len(record.diagnostics) > 1:
            raise PrivateReportError("an event fate is ambiguous")
        # Scoring consumes only the outcome. The ledger record keeps the fate's
        # evidence chain, so the stage and record-ID tuples are left empty.
        fates[record.artifact_id] = EventFate(
            event_id=record.artifact_id,
            arm_id=arm_id,
            outcome=outcome,
            first_missing_stage=None,
            reason=record.diagnostics[0].value if record.diagnostics else outcome.value,
            evidence_record_ids=(),
            diagnostic_record_ids=(),
        )
    if fates.keys() != {event.event_id for event in fixture.events}:
        raise PrivateReportError("every reference event needs exactly one arm fate")
    return fates


def _raw_nonblank(
    event: ReferenceEvent,
    outcome: TerminalOutcome,
    alignment: EventAlignment | None,
    hypotheses: Mapping[str, ScoredHypothesis],
) -> bool | None:
    if not event_is_eligible(event, outcome) or outcome is TerminalOutcome.UNKNOWN:
        return None
    if alignment is None:
        # An unresolved component assigns no hypothesis to its events.
        return None
    return any(hypotheses[item].text.strip() for item in alignment.hypothesis_ids)


def _resources(fixture: _Fixture, arm_id: str) -> ResourceObservations:
    measurements: dict[object, EvidenceRecord] = {}
    for record in fixture.records:
        if record.arm_id != arm_id or record.kind is not RecordKind.MEASUREMENT:
            continue
        name = record.details.get("measurement_name")
        if name in measurements:
            raise PrivateReportError("an arm measurement is repeated")
        measurements[name] = record
    observations: dict[str, ScalarObservation] = {}
    for name, unit in _RESOURCE_UNITS:
        record = measurements.get(name)
        if record is None:
            raise PrivateReportError("an arm resource measurement is missing")
        observed = record.observation_status is ObservationStatus.OBSERVED
        observations[name] = ScalarObservation(
            record.observation_status,
            cast(int | float, record.details["value"]) if observed else None,
            unit,
        )
    return ResourceObservations(**observations)


def _decoder_windows(
    fixture: _Fixture,
    kind: ArmKind,
) -> tuple[ObservationStatus, tuple[SampleSpan, ...]]:
    if kind not in _REPLAY_ARM_KINDS:
        # P1 decodes oracle slices, not replay windows.
        return ObservationStatus.OUT_OF_SCOPE, ()
    windows = tuple(
        record
        for record in fixture.records
        if record.kind is RecordKind.WINDOW
        and record.observation_status is ObservationStatus.OBSERVED
    )
    if not windows:
        return ObservationStatus.INSTRUMENTATION_MISSING, ()
    return (
        ObservationStatus.OBSERVED,
        tuple(span for record in windows for span in record.source_spans),
    )


def _vad_activation(
    fixture: _Fixture,
    kind: ArmKind,
) -> tuple[ObservationStatus, tuple[SampleSpan, ...]]:
    """Merge the frame spans where the replayed frame gate was in speech."""

    if kind not in _REPLAY_ARM_KINDS:
        # P1 decodes oracle slices, so no VAD gate decides its windows.
        return ObservationStatus.OUT_OF_SCOPE, ()
    frames = tuple(
        record
        for record in fixture.records
        if record.kind is RecordKind.FRAME
        and record.observation_status is ObservationStatus.OBSERVED
    )
    if not frames:
        return ObservationStatus.INSTRUMENTATION_MISSING, ()
    positive: list[SampleSpan] = []
    for record in frames:
        in_speech = record.details.get("gate_in_speech_after")
        if type(in_speech) is not bool:
            raise PrivateReportError("a frame record has no gate speech state")
        if in_speech:
            positive.extend(record.source_spans)
    merged: list[SampleSpan] = []
    for span in sorted(positive, key=lambda item: (item.timeline_id, item.start)):
        last = merged[-1] if merged else None
        if last is not None and last.timeline_id == span.timeline_id and span.start <= last.end:
            merged[-1] = SampleSpan(last.timeline_id, last.start, max(last.end, span.end))
        else:
            merged.append(span)
    return ObservationStatus.OBSERVED, tuple(merged)


def _hallucination_ids(
    fixture: _Fixture,
    accepted: Iterable[ScoredHypothesis],
) -> tuple[str, ...]:
    """Accepted output inside a non-speech control and outside every event."""

    if not fixture.non_speech_control_spans:
        return ()
    hallucinations: list[str] = []
    for hypothesis in accepted:
        spans = tuple(
            span
            for span in hypothesis.source_spans
            if span.timeline_id == fixture.timeline_id
        )
        if not spans:
            continue
        start = min(span.start for span in spans)
        end = max(span.end for span in spans)
        if any(
            max(start, control.start) < min(end, control.end)
            for control in fixture.non_speech_control_spans
        ) and not any(
            max(start, event.span.start) < min(end, event.span.end)
            for event in fixture.events
        ):
            hallucinations.append(hypothesis.hypothesis_id)
    return tuple(hallucinations)


def _vocal_sounds(
    events: Iterable[ScoredEvent],
    accepted: Iterable[ScoredHypothesis],
) -> VocalSoundCounts:
    """Count what the words level removes from references and accepted output."""

    references = tuple(
        split_vocal_sounds(event.text) for event in events if event.text is not None
    )
    outputs = tuple(split_vocal_sounds(hypothesis.text) for hypothesis in accepted)
    return VocalSoundCounts(
        reference_sound_characters=sum(item.sound_characters for item in references),
        output_sound_characters=sum(item.sound_characters for item in outputs),
        reference_chains=sum(item.chains for item in references),
        output_chains=sum(item.chains for item in outputs),
    )


def _score_fixture(fixture: _Fixture, arm_id: str, kind: ArmKind) -> ScoredFixture:
    hypotheses, decoder_calls = _raw_hypotheses(fixture, arm_id)
    hypotheses_by_id = {item.hypothesis_id: item for item in hypotheses}
    accepted_ids = _wrapper_accepted_ids(fixture, arm_id, hypotheses_by_id)
    alignments = _normalized_alignments(
        fixture.scored_events,
        tuple(hypotheses_by_id[item] for item in accepted_ids),
    )
    accepted_minimal = _event_alignments(
        next(
            item for item in alignments if item.level is NormalizationLevel.MINIMAL
        ).components
    )
    raw_minimal = _event_alignments(
        align_component(component, JA_TIME_DP_1_0, normalize_minimal)
        for component in build_temporal_components(
            fixture.scored_events,
            hypotheses,
            JA_TIME_DP_1_0,
        )
    )
    fates = _event_fates(fixture, arm_id)
    truncated_ids = {
        record.artifact_id
        for record in _observed(fixture, arm_id, RecordKind.DIAGNOSTIC)
        if record.decision == DiagnosticCode.ENDPOINT_BOUNDARY_TRUNCATED.value
    }
    window_status, window_spans = _decoder_windows(fixture, kind)
    vad_status, vad_spans = _vad_activation(fixture, kind)
    hard_tags = set(fixture.hard_stratum_tags)
    return ScoredFixture(
        fixture=fixture.manifest,
        arm_id=arm_id,
        timeline_id=fixture.timeline_id,
        sample_rate_hz=fixture.sample_rate_hz,
        sample_count=fixture.sample_count,
        model_runtime_sha256=fixture.source.model_runtime_sha256,
        events=tuple(
            EventScore(
                reference=event,
                fate=fates[event.event_id],
                raw_nonblank=_raw_nonblank(
                    event,
                    fates[event.event_id].outcome,
                    raw_minimal.get(event.event_id),
                    hypotheses_by_id,
                ),
                lexically_usable=event_is_lexically_usable(
                    event,
                    fates[event.event_id].outcome,
                    accepted_minimal.get(event.event_id),
                    JA_EVENT_USABLE_1_1,
                ),
                paced_deadline_met=None,
                output_latency_seconds=_UNPACED_LATENCY,
            )
            for event in fixture.events
        ),
        normalized_alignments=alignments,
        expected_wrapper_hypothesis_ids=accepted_ids,
        vad_observation_status=vad_status,
        vad_positive_spans=vad_spans,
        decoder_window_observation_status=window_status,
        decoder_window_spans=window_spans,
        boundary_truncated_event_ids=tuple(
            event.event_id for event in fixture.events if event.event_id in truncated_ids
        ),
        duplicate_event_ids=duplicate_event_ids(
            (item for item in accepted_minimal.values() if item is not None),
            decoder_calls,
        ),
        duplicate_observation_status=ObservationStatus.OBSERVED,
        hard_event_ids=tuple(
            event.event_id for event in fixture.events if hard_tags & set(event.tags)
        ),
        hard_stratum_version=fixture.hard_stratum_version,
        hard_stratum_sha256=canonical_sha256(
            {
                "version": fixture.hard_stratum_version,
                "tags": list(fixture.hard_stratum_tags),
            }
        ),
        non_speech_control_spans=fixture.non_speech_control_spans,
        hallucination_output_ids=_hallucination_ids(
            fixture,
            (hypotheses_by_id[item] for item in accepted_ids),
        ),
        resources=_resources(fixture, arm_id),
        vocal_sounds=_vocal_sounds(
            fixture.scored_events,
            (hypotheses_by_id[item] for item in accepted_ids),
        ),
    )


def _has_stochastic_fallback(fixture: _Fixture, arm_id: str) -> bool:
    return any(
        DiagnosticCode.DECODER_STOCHASTIC_FALLBACK in record.diagnostics
        for record in _observed(fixture, arm_id, RecordKind.RAW_HYPOTHESIS)
    )


def score_runs(
    run_dirs: Sequence[Path],
    boundary: PrivateBoundary,
    *,
    bootstrap_replicates: int = 10_000,
) -> RunScores:
    """Score every arm complete in all cohort runs, once over all fixtures."""

    sources = verify_report_sources(run_dirs, boundary)
    fixtures = tuple(_load_fixture(source, boundary) for source in sources)
    complete_arm_ids = set.intersection(
        *(
            {
                status.arm_id
                for status in source.integrity.computed_arm_statuses
                if status.status is ArmStatus.COMPLETE
            }
            for source in sources
        )
    )
    run_ids = tuple(source.start.run_id for source in sources)
    reports: list[ScoreReport] = []
    omitted: list[str] = []
    for definition in sources[0].start.arm_definitions:
        if definition.arm_id not in complete_arm_ids:
            continue
        if definition.kind is ArmKind.P2B and any(
            _has_stochastic_fallback(fixture, definition.arm_id) for fixture in fixtures
        ):
            # Segments flagged decoder_stochastic_fallback are excluded from
            # P2b interpretation, so the whole arm stays out of the bundle.
            omitted.append(definition.arm_id)
            continue
        reports.append(
            compute_metrics(
                tuple(
                    _score_fixture(fixture, definition.arm_id, definition.kind)
                    for fixture in fixtures
                ),
                bootstrap_replicates=bootstrap_replicates,
            )
        )
    if not reports:
        raise PrivateReportError("no complete arm can be scored")
    return RunScores(
        reports=tuple(reports),
        score_sources={report.arm_id: run_ids for report in reports},
        omitted_stochastic_arm_ids=tuple(omitted),
    )


def write_run_scores(
    run_dirs: Sequence[Path],
    boundary: PrivateBoundary,
    *,
    bootstrap_replicates: int = 10_000,
) -> ScoreBundle:
    """Score the runs and persist them as a new ``reports/<uuid4>`` bundle."""

    scores = score_runs(
        run_dirs,
        boundary,
        bootstrap_replicates=bootstrap_replicates,
    )
    report_id = new_opaque_id()
    bundle_path = write_score_bundle(
        boundary.root / "reports" / report_id,
        run_dirs,
        scores.reports,
        scores.score_sources,
        boundary,
    )
    return ScoreBundle(
        report_id=report_id,
        bundle_path=bundle_path,
        omitted_stochastic_arm_ids=scores.omitted_stochastic_arm_ids,
    )


__all__ = (
    "RunScores",
    "ScoreBundle",
    "score_runs",
    "write_run_scores",
)
