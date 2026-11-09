"""Paired candidate-versus-baseline contrasts over title-clustered score reports."""

from __future__ import annotations

from dataclasses import replace

from scripts.e0_evidence.align import ScoredEvent
from scripts.e0_evidence.compare import paired_contrasts
from scripts.e0_evidence.schema import ObservationStatus, TerminalOutcome
from scripts.e0_evidence.score import (
    CerMetric,
    EventScore,
    MetricStatus,
    RateMetric,
    ScalarObservation,
    ScoreReport,
    compute_metrics,
)
from tests.e0_evidence.test_scoring import (
    _id,
    _plain_fate,
    _plain_fixture,
    _reference_event,
)


_BASELINE = ((6, 10), (3, 10), (8, 10), (5, 10))
_CANDIDATE_REPORT_ID = "96000000-0000-4000-8000-000000000001"


def _four_title_report() -> ScoreReport:
    arm_id = _id(560_000)
    latency = ScalarObservation(ObservationStatus.OBSERVED, 0.25, "seconds")
    fixtures = []
    for title_index in range(len(_BASELINE)):
        reference = _reference_event(2_000 + title_index, 0, 8_000)
        event = EventScore(
            reference,
            _plain_fate(reference, arm_id, TerminalOutcome.NO_DECODER_WINDOW),
            raw_nonblank=False,
            lexically_usable=False,
            paced_deadline_met=False,
            output_latency_seconds=latency,
        )
        fixtures.append(
            _plain_fixture(
                90 + title_index,
                arm_id,
                16_000,
                (event,),
                (ScoredEvent(reference.event_id, reference.span, "猫"),),
                (),
            )
        )
    return compute_metrics(tuple(fixtures), bootstrap_replicates=0)


def _with_title_counts(
    report: ScoreReport,
    counts: tuple[tuple[int, int], ...],
) -> ScoreReport:
    """Set each title's recalls to ``successes / events`` and CER to the misses."""

    titles = []
    for title, (successes, events) in zip(report.title_scores, counts, strict=True):
        recall = successes / events
        rate = RateMetric(
            successes, events, 0, MetricStatus.DEFINED, recall, recall, recall
        )
        cer = CerMetric(
            substitutions=events - successes,
            deletions=0,
            insertions=0,
            reference_characters=events,
            unresolved_events=0,
            unresolved_components=0,
            unassigned_insertions=0,
            status=MetricStatus.DEFINED,
            value=(events - successes) / events,
        )
        titles.append(
            replace(
                title,
                metrics=replace(
                    title.metrics,
                    lexically_usable_recall=rate,
                    wrapper_available_recall=rate,
                    cer_minimal=cer,
                ),
            )
        )
    return replace(report, title_scores=tuple(titles))


def test_identical_candidate_has_zero_delta_and_is_not_significant() -> None:
    baseline = _with_title_counts(_four_title_report(), _BASELINE)

    contrasts = paired_contrasts(
        baseline,
        {_CANDIDATE_REPORT_ID: baseline},
        confirmatory=True,
        replicates=2_000,
    )

    assert [item.metric for item in contrasts] == [
        "lexically_usable_recall",
        "wrapper_available_recall",
        "cer_minimal",
    ]
    for item in contrasts:
        assert item.status == "confirmatory"
        assert item.delta == 0.0
        assert item.ci_lower is not None and item.ci_upper is not None
        assert item.ci_lower <= 0.0 <= item.ci_upper
        assert item.significant is False


def test_candidate_better_on_every_title_has_a_positive_recall_interval() -> None:
    report = _four_title_report()
    baseline = _with_title_counts(report, _BASELINE)
    candidate = _with_title_counts(report, ((8, 10), (5, 10), (9, 10), (7, 10)))

    contrasts = {
        item.metric: item
        for item in paired_contrasts(
            baseline,
            {_CANDIDATE_REPORT_ID: candidate},
            confirmatory=True,
            replicates=2_000,
        )
    }

    for metric in ("lexically_usable_recall", "wrapper_available_recall"):
        assert contrasts[metric].delta is not None and contrasts[metric].delta > 0
        assert contrasts[metric].ci_lower is not None
        assert contrasts[metric].ci_lower > 0
        assert contrasts[metric].significant is True
    # Fewer character errors is the better direction for CER.
    assert contrasts["cer_minimal"].ci_upper is not None
    assert contrasts["cer_minimal"].ci_upper < 0
