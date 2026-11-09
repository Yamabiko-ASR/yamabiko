"""Paired candidate-versus-baseline contrasts over verified E0 score bundles.

Every candidate is scored as its own cohort over the same fixtures as the
baseline. A contrast pools each metric over titles, and its interval comes from
a title-cluster bootstrap in which the baseline and every candidate share one
draw matrix, so the resampled titles are always paired.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from scripts.e0_evidence._common import exclusive_write_bytes
from scripts.e0_evidence.private_root import PrivateBoundary, require_private_path
from scripts.e0_evidence.report import (
    PrivateReportError,
    ReportSet,
    load_verified_score_bundle,
)
from scripts.e0_evidence.schema import (
    ArmKind,
    Partition,
    PrivateContentClass,
    new_opaque_id,
)
from scripts.e0_evidence.score import (
    AggregateMetrics,
    CerMetric,
    MetricStatus,
    RateMetric,
    ScoreReport,
    percentile_type7,
)
from scripts.e0_evidence.wire import canonical_json_bytes, inherited_classification


COMPARISON_SCHEMA_VERSION = "e0-comparison-1.0"
CONTRAST_METRICS = ("lexically_usable_recall", "wrapper_available_recall", "cer_minimal")
_SIGNIFICANCE_LEVEL = 0.05
_RUN_START_BINDINGS = (
    "reference_set_id",
    "reference_set_sha256",
    "reference_text_sha256",
    "master_pcm_sha256",
)


@dataclass(frozen=True, slots=True)
class Contrast:
    """One candidate-minus-baseline difference for one pooled metric.

    ``status`` is ``confirmatory`` (Holm-adjusted across every complete
    contrast), ``exploratory`` (unadjusted), or ``incomplete`` (some title
    metric was not defined, so no value or inference is reported).
    """

    candidate_report_id: str
    metric: str
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None
    ci_lower: float | None
    ci_upper: float | None
    p_value: float | None
    holm_p: float | None
    significant: bool | None
    status: str


def _title_counts(
    report: ScoreReport,
    metric: str,
) -> tuple[dict[str, tuple[int, int]], bool]:
    """Return per-title (errors or successes, denominator) and completeness."""

    counts: dict[str, tuple[int, int]] = {}
    complete = True
    for title in report.title_scores:
        value = getattr(title.metrics, metric)
        if value.status is not MetricStatus.DEFINED:
            complete = False
        if isinstance(value, RateMetric):
            if value.unknown > 0:
                complete = False
            counts[title.title_id] = (value.numerator, value.denominator)
        elif isinstance(value, CerMetric):
            counts[title.title_id] = (
                value.substitutions + value.deletions + value.insertions,
                value.reference_characters,
            )
        else:  # pragma: no cover - CONTRAST_METRICS names only these types
            raise TypeError(f"{metric} is not a rate or CER metric")
    return counts, complete


def _pooled(
    counts: Mapping[str, tuple[int, int]],
    titles: Sequence[str],
) -> float | None:
    numerator = sum(counts[title][0] for title in titles)
    denominator = sum(counts[title][1] for title in titles)
    return None if denominator == 0 else numerator / denominator


def _title_ids(report: ScoreReport) -> tuple[str, ...]:
    return tuple(sorted(title.title_id for title in report.title_scores))


def _holm(p_values: Sequence[float]) -> tuple[float, ...]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return tuple(adjusted)


def paired_contrasts(
    baseline: ScoreReport,
    candidates: Mapping[str, ScoreReport],
    *,
    confirmatory: bool,
    seed: int = 105,
    replicates: int = 10_000,
) -> tuple[Contrast, ...]:
    """Compare each candidate with the baseline on every contrast metric."""

    if not isinstance(baseline, ScoreReport):
        raise TypeError("baseline must be ScoreReport")
    if not candidates:
        raise ValueError("at least one candidate is required")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    titles = _title_ids(baseline)
    if any(_title_ids(report) != titles for report in candidates.values()):
        raise ValueError("candidates must score the same titles as the baseline")
    rng = random.Random(seed)
    draws = tuple(
        tuple(titles[rng.randrange(len(titles))] for _ in titles)
        for _ in range(replicates)
    )

    drafts: list[dict[str, object]] = []
    for candidate_id, candidate in sorted(candidates.items()):
        for metric in CONTRAST_METRICS:
            base_counts, base_complete = _title_counts(baseline, metric)
            cand_counts, cand_complete = _title_counts(candidate, metric)
            draft: dict[str, object] = {
                "candidate_report_id": candidate_id,
                "metric": metric,
                "baseline_value": None,
                "candidate_value": None,
                "delta": None,
                "ci_lower": None,
                "ci_upper": None,
                "p_value": None,
            }
            base_value = _pooled(base_counts, titles)
            cand_value = _pooled(cand_counts, titles)
            deltas: list[float] = []
            if base_complete and cand_complete and None not in (base_value, cand_value):
                for draw in draws:
                    base_draw = _pooled(base_counts, draw)
                    cand_draw = _pooled(cand_counts, draw)
                    if base_draw is None or cand_draw is None:
                        continue
                    deltas.append(cand_draw - base_draw)
            if deltas:
                valid = len(deltas)
                at_or_below = sum(delta <= 0 for delta in deltas)
                at_or_above = sum(delta >= 0 for delta in deltas)
                draft.update(
                    baseline_value=base_value,
                    candidate_value=cand_value,
                    delta=cand_value - base_value,  # type: ignore[operator]
                    ci_lower=percentile_type7(deltas, 0.025),
                    ci_upper=percentile_type7(deltas, 0.975),
                    p_value=min(
                        1.0,
                        2
                        * min(
                            (1 + at_or_below) / (valid + 1),
                            (1 + at_or_above) / (valid + 1),
                        ),
                    ),
                )
            drafts.append(draft)

    complete = [index for index, draft in enumerate(drafts) if draft["p_value"] is not None]
    adjusted = (
        dict(
            zip(
                complete,
                _holm([drafts[index]["p_value"] for index in complete]),  # type: ignore[misc]
            )
        )
        if confirmatory
        else {}
    )
    contrasts: list[Contrast] = []
    for index, draft in enumerate(drafts):
        p_value = draft["p_value"]
        if p_value is None:
            contrasts.append(
                Contrast(**draft, holm_p=None, significant=None, status="incomplete")  # type: ignore[arg-type]
            )
            continue
        holm_p = adjusted.get(index)
        tested = holm_p if confirmatory else p_value
        contrasts.append(
            Contrast(
                **draft,  # type: ignore[arg-type]
                holm_p=holm_p,
                significant=tested < _SIGNIFICANCE_LEVEL,  # type: ignore[operator]
                status="confirmatory" if confirmatory else "exploratory",
            )
        )
    return tuple(contrasts)


def _arm_report(report_set: ReportSet, arm_kind: ArmKind) -> ScoreReport:
    arm_ids = [
        mapping["arm_id"]
        for mapping in report_set.score_sources
        if mapping["arm_kind"] == arm_kind.value
    ]
    if len(arm_ids) != 1:
        raise ValueError(
            f"score bundle {report_set.report_id} has no single {arm_kind.value} report"
        )
    (report,) = (item for item in report_set.score_reports if item.arm_id == arm_ids[0])
    return report


def _fixture_bindings(report_set: ReportSet) -> dict[str, tuple[object, ...]]:
    bindings: dict[str, tuple[object, ...]] = {}
    for source in report_set.sources:
        binding = tuple(source.start.details.get(key) for key in _RUN_START_BINDINGS)
        if bindings.setdefault(source.fixture_id, binding) != binding:
            raise ValueError("a score bundle binds one fixture to two references")
    return bindings


def _format(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown(payload: Mapping[str, object], contrasts: Sequence[Contrast]) -> str:
    lines = [
        "# Yamabiko E0 paired comparison",
        "",
        f"Comparison ID: `{payload['comparison_id']}`",
        f"Baseline report: `{payload['baseline_report_id']}`",
        f"Arm kind: `{payload['arm_kind']}`",
        f"Partition: `{payload['partition']}`",
        f"Bootstrap: seed `{payload['bootstrap_seed']}`, "
        f"`{payload['bootstrap_replicates']}` title-cluster replicates",
        "",
        "| Candidate report | Metric | Baseline | Candidate | Delta | 95% CI | p | "
        "Holm p | Significant | Status |",
        "|---|---|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for item in contrasts:
        interval = (
            "n/a"
            if item.ci_lower is None
            else f"[{_format(item.ci_lower)}, {_format(item.ci_upper)}]"
        )
        lines.append(
            f"| {item.candidate_report_id} | {item.metric} | "
            f"{_format(item.baseline_value)} | {_format(item.candidate_value)} | "
            f"{_format(item.delta)} | {interval} | {_format(item.p_value)} | "
            f"{_format(item.holm_p)} | {_format(item.significant)} | {item.status} |"
        )
    return "\n".join(lines) + "\n"


def compare_reports(
    baseline_report_id: str,
    candidate_report_ids: Sequence[str],
    boundary: PrivateBoundary,
    *,
    arm_kind: ArmKind = ArmKind.P0,
    replicates: int = 10_000,
    seed: int = 105,
) -> Path:
    """Write a paired comparison under ``reports/<uuid4>`` and return that directory."""

    candidate_ids = tuple(candidate_report_ids)
    if not candidate_ids:
        raise ValueError("at least one candidate report is required")
    if len(set(candidate_ids)) != len(candidate_ids) or baseline_report_id in candidate_ids:
        raise ValueError("baseline and candidate reports must be distinct")
    bundles = {
        report_id: load_verified_score_bundle(
            boundary.root / "reports" / report_id,
            boundary,
        )
        for report_id in (baseline_report_id, *candidate_ids)
    }
    reports = {
        report_id: _arm_report(bundle, arm_kind) for report_id, bundle in bundles.items()
    }
    baseline_bundle = bundles[baseline_report_id]
    baseline = reports[baseline_report_id]
    bindings = _fixture_bindings(baseline_bundle)
    for report_id in candidate_ids:
        bundle = bundles[report_id]
        report = reports[report_id]
        if bundle.partition is not baseline_bundle.partition:
            raise ValueError("compared bundles must share a partition")
        if bundle.cohort_fixture_ids != baseline_bundle.cohort_fixture_ids:
            raise ValueError("compared bundles must score the same fixtures")
        if _title_ids(report) != _title_ids(baseline):
            raise ValueError("compared bundles must score the same titles")
        if report.scorer_source_sha256 != baseline.scorer_source_sha256:
            raise ValueError("compared bundles were scored by different scorer sources")
        if _fixture_bindings(bundle) != bindings:
            raise ValueError(
                "compared bundles must bind every fixture to the same master and reference"
            )
    confirmatory = baseline_bundle.partition is Partition.LOCKED_CONFIRMATION
    contrasts = paired_contrasts(
        baseline,
        {report_id: reports[report_id] for report_id in candidate_ids},
        confirmatory=confirmatory,
        seed=seed,
        replicates=replicates,
    )
    comparison_id = new_opaque_id()
    payload: dict[str, object] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "classification": inherited_classification(
            PrivateContentClass.INTERNAL_AGGREGATE,
            (bundle.classification for bundle in bundles.values()),
        ).value,
        "comparison_id": comparison_id,
        "baseline_report_id": baseline_report_id,
        "candidate_report_ids": list(candidate_ids),
        "arm_kind": arm_kind.value,
        "partition": baseline_bundle.partition.value,
        "confirmatory": confirmatory,
        "cohort_fixture_ids": list(baseline_bundle.cohort_fixture_ids),
        "title_ids": list(_title_ids(baseline)),
        "scorer_source_sha256": baseline.scorer_source_sha256,
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "significance_level": _SIGNIFICANCE_LEVEL,
        "contrasts": [
            {
                "candidate_report_id": item.candidate_report_id,
                "metric": item.metric,
                "baseline_value": item.baseline_value,
                "candidate_value": item.candidate_value,
                "delta": item.delta,
                "ci_lower": item.ci_lower,
                "ci_upper": item.ci_upper,
                "p_value": item.p_value,
                "holm_p": item.holm_p,
                "significant": item.significant,
                "status": item.status,
            }
            for item in contrasts
        ],
    }
    json_bytes = canonical_json_bytes(payload, newline=True)  # type: ignore[arg-type]
    markdown_bytes = _markdown(payload, contrasts).encode("utf-8")
    reports_root = require_private_path(boundary.root / "reports", boundary)
    comparison_dir = require_private_path(reports_root / comparison_id, boundary)
    try:
        comparison_dir.mkdir(exist_ok=False)
    except OSError:
        raise PrivateReportError("comparison directory reservation failed") from None
    created: list[Path] = []
    try:
        for name, data in (
            ("comparison.private.md", markdown_bytes),
            ("comparison.private.json", json_bytes),
        ):
            path = require_private_path(comparison_dir / name, boundary)
            exclusive_write_bytes(path, data)
            created.append(path)
    except (OSError, UnicodeError):
        for path in reversed(created):
            path.unlink(missing_ok=True)
        comparison_dir.rmdir()
        raise PrivateReportError("comparison write failed") from None
    return comparison_dir


__all__ = (
    "COMPARISON_SCHEMA_VERSION",
    "CONTRAST_METRICS",
    "Contrast",
    "compare_reports",
    "paired_contrasts",
)
