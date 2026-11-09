"""Behavior checks for frozen Japanese normalization and temporal alignment."""

from __future__ import annotations

import math
from types import SimpleNamespace

from scripts.e0_evidence import runtime
from scripts.e0_evidence.align import (
    AlignmentStatus,
    ComponentAlignment,
    EditKind,
    EventAlignment,
    JA_TIME_DP_1_0,
    ScoredEvent,
    ScoredHypothesis,
    align_component,
    build_temporal_components,
    character_trace,
)
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
from scripts.e0_evidence.references import ReferenceEvent, VerbatimReference
from scripts.e0_evidence.schema import (
    ArmKind,
    ArmStatus,
    DiagnosticCode,
    EvidenceRecord,
    EvidenceRecordDraft,
    EventFate,
    FixtureManifest,
    MONOTONIC_CLOCK_ID,
    ObservationStatus,
    Partition,
    RecordKind,
    SCHEMA_VERSION,
    SampleSpan,
    Stage,
    TerminalOutcome,
)
from scripts.e0_evidence.score import (
    JA_EVENT_USABLE_1_1,
    RAW_UNICODE_1_0_SHA256,
    RAW_UNICODE_1_0_VERSION,
    ArmEvidence,
    EventScore,
    MetricStatus,
    NormalizationLevel,
    NormalizedAlignmentSet,
    ResourceObservations,
    ScalarObservation,
    ScoredFixture,
    VocalSoundCounts,
    compute_metrics,
    derive_event_fates,
    duplicate_event_ids,
    event_is_lexically_usable,
    scored_reference_events,
)


_TIMELINE_ID = "91000000-0000-4000-8000-000000000001"
# The scored fixtures' texts hold no laughs, breaths, or interjection chains.
_NO_VOCAL_SOUNDS = VocalSoundCounts(0, 0, 0, 0)


def _id(ordinal: int) -> str:
    return f"91000000-0000-4000-8001-{ordinal:012d}"


def _event(ordinal: int, start: int, end: int, text: str) -> ScoredEvent:
    return ScoredEvent(
        event_id=_id(ordinal),
        span=SampleSpan(_TIMELINE_ID, start, end),
        text=text,
    )


def _hypothesis(
    ordinal: int,
    start: int,
    end: int,
    text: str,
) -> ScoredHypothesis:
    return ScoredHypothesis(
        hypothesis_id=_id(ordinal),
        source_spans=(SampleSpan(_TIMELINE_ID, start, end),),
        text=text,
    )


def _reference_event(
    ordinal: int,
    start: int,
    end: int,
    *,
    text_reference: bool = True,
) -> ReferenceEvent:
    return ReferenceEvent(
        event_id=_id(ordinal),
        master_id=_id(900_001),
        span=SampleSpan(_TIMELINE_ID, start, end),
        intelligibility="intelligible",
        verbatim_ref_id=_id(600_000 + ordinal) if text_reference else None,
        tags=(),
        annotation_version="test-reference-1.0",
    )


def _record(
    ordinal: int,
    *,
    arm_id: str | None,
    stage: Stage,
    kind: RecordKind,
    source_spans: tuple[SampleSpan, ...] = (),
    parent_ids: tuple[str, ...] = (),
    decision: str | None = None,
    diagnostics: tuple[DiagnosticCode, ...] = (),
    details: dict[str, object] | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        schema_version=SCHEMA_VERSION,
        record_id=_id(700_000 + ordinal),
        run_id=_id(900_002),
        arm_id=arm_id,
        session_id=_id(900_003),
        seq=ordinal,
        observed_monotonic_ns=ordinal,
        clock_id=MONOTONIC_CLOCK_ID,
        stage=stage,
        kind=kind,
        artifact_id=_id(800_000 + ordinal),
        parent_ids=parent_ids,
        source_spans=source_spans,
        decision=decision,
        diagnostics=diagnostics,
        details={} if details is None else details,
    )


class _RecordingFateSink:
    def __init__(self) -> None:
        self.records: list[EvidenceRecord] = []

    def emit(self, draft: EvidenceRecordDraft) -> EvidenceRecord:
        ordinal = 950_000 + len(self.records)
        record = EvidenceRecord(
            schema_version=SCHEMA_VERSION,
            record_id=_id(ordinal),
            run_id=_id(900_002),
            arm_id=draft.arm_id,
            session_id=_id(900_004),
            seq=len(self.records),
            observed_monotonic_ns=len(self.records),
            clock_id=MONOTONIC_CLOCK_ID,
            stage=draft.stage,
            kind=draft.kind,
            artifact_id=draft.artifact_id,
            parent_ids=draft.parent_ids,
            source_spans=draft.source_spans,
            observation_status=draft.observation_status,
            decision=draft.decision,
            diagnostics=draft.diagnostics,
            classification=draft.classification,
            details=draft.details,
        )
        self.records.append(record)
        return record


def _alignments(
    events: tuple[ScoredEvent, ...],
    hypotheses: tuple[ScoredHypothesis, ...],
    normalizer=normalize_minimal,
) -> tuple[ComponentAlignment, ...]:
    return tuple(
        align_component(component, JA_TIME_DP_1_0, normalizer)
        for component in build_temporal_components(
            events,
            hypotheses,
            JA_TIME_DP_1_0,
        )
    )


def _event_alignment(
    event_id: str,
    hypothesis_id: str,
    reference: str,
    hypothesis: str,
) -> EventAlignment:
    trace = character_trace(reference, hypothesis)
    return EventAlignment(
        event_id=event_id,
        hypothesis_ids=(hypothesis_id,),
        substitutions=trace.substitutions,
        deletions=trace.deletions,
        insertions=trace.insertions,
        operations=trace.operations,
    )


def _normalization_set(
    level: NormalizationLevel,
    events: tuple[ScoredEvent, ...],
    hypotheses: tuple[ScoredHypothesis, ...],
) -> NormalizedAlignmentSet:
    if level is NormalizationLevel.RAW:
        version = RAW_UNICODE_1_0_VERSION
        sha256 = RAW_UNICODE_1_0_SHA256
        normalizer = normalize_raw
    elif level is NormalizationLevel.MINIMAL:
        version = JA_MINIMAL_1_1.version
        sha256 = JA_MINIMAL_1_1.sha256
        normalizer = normalize_minimal
    elif level is NormalizationLevel.LENIENT:
        version = JA_LENIENT_1_1.version
        sha256 = JA_LENIENT_1_1.sha256
        normalizer = normalize_lenient
    else:
        version = JA_WORDS_1_0.version
        sha256 = JA_WORDS_1_0.sha256
        normalizer = normalize_words
    return NormalizedAlignmentSet(
        level=level,
        normalization_policy_version=version,
        normalization_policy_sha256=sha256,
        alignment_policy_version=JA_TIME_DP_1_0.version,
        alignment_policy_sha256=JA_TIME_DP_1_0.sha256,
        components=_alignments(events, hypotheses, normalizer),
    )


def test_frozen_normalization_is_symmetric_and_character_trace_preserves_direction() -> (
    None
):
    raw = "　Ａ、うん カナ猫１２ー！"
    normalized = "Aうんカナ猫12ー"

    assert normalize_raw(raw) == raw
    assert normalize_minimal(raw) == normalized
    assert normalize_lenient(raw) == "aうんかな猫12ー"

    event = _event(1, 0, 8_000, raw)
    hypothesis = _hypothesis(101, 0, 8_000, normalized)
    component = build_temporal_components(
        (event,),
        (hypothesis,),
        JA_TIME_DP_1_0,
    )[0]

    alignment = align_component(component, JA_TIME_DP_1_0, normalize_minimal)

    assert alignment.status is AlignmentStatus.RESOLVED
    assert (
        alignment.substitutions,
        alignment.deletions,
        alignment.insertions,
    ) == (0, 0, 0)

    forward = character_trace("うん猫", "うん犬ね")
    reverse = character_trace("うん犬ね", "うん猫")

    assert (forward.substitutions, forward.deletions, forward.insertions) == (
        1,
        0,
        1,
    )
    assert tuple(operation.kind for operation in forward.operations) == (
        EditKind.MATCH,
        EditKind.MATCH,
        EditKind.SUBSTITUTION,
        EditKind.INSERTION,
    )
    assert (reverse.substitutions, reverse.deletions, reverse.insertions) == (
        1,
        1,
        0,
    )
    assert tuple(operation.kind for operation in reverse.operations) == (
        EditKind.MATCH,
        EditKind.MATCH,
        EditKind.SUBSTITUTION,
        EditKind.DELETION,
    )


def test_minimal_normalization_removes_symbol_only_text() -> None:
    # Tildes, music notes, and equals signs are Unicode symbols (S*), not
    # speech; leaving them would charge transcripts for decoration.
    assert normalize_minimal("〜～~♪＝") == ""


def test_words_normalization_removes_laughs_breaths_and_chain_repeats_but_keeps_words() -> (
    None
):
    # (text, words left, minimal characters removed, chains)
    cases = (
        *(
            (sound, "", len(sound), 0)
            for sound in (
                "ははは",
                "あはは",
                "えへへ",
                "うふふ",
                "ふふっ",
                "ひひひ",
                "はぁ",
                "はあ",
                "ふう",
                "すう",
                "ひい",
                "ほお",
                "ほう",
                "すー",
                "すーすー",
                "はぁはぁ",
            )
        ),
        ("えっえっえっえっ", "えっ", 6, 1),
        *(
            (word, word, 0, 0)
            for word in (
                "うん",
                "はい",
                "え",
                "えっ",
                "ええ",
                "あの",
                "えーと",
                "ん",
                "ふん",
                "あっ",
                "うんうん",
                "スーパーに行く",
                "母は元気",
            )
        ),
        # ふう is a noun here (like this), so the content-word guard keeps it.
        ("こういうふうに", "こういうふうに", 0, 0),
        # A drawn-out particle は is speech; a laugh after one is still a laugh.
        ("今日はー、いい天気", "今日はーいい天気", 0, 0),
        ("それはあ、違う", "それはあ違う", 0, 0),
        ("彼はああ言った", "彼はああ言った", 0, 0),
        ("そうなんだ、ははは", "そうなんだ", 3, 0),
        # A breath may draw out several vowels or mix units.
        ("はぁぁ、疲れた", "疲れた", 3, 0),
        ("ひいい、怖い", "怖い", 3, 0),
        ("ふうう", "", 3, 0),
        ("すー、はー、落ち着いて", "落ち着いて", 4, 0),
        ("すーはーすーはー", "", 8, 0),
        # A drawn-out です is speech.
        ("なんですう", "なんですう", 0, 0),
        ("好きですう", "好きですう", 0, 0),
        # Katakana is folded only to match; the words keep the original text.
        ("そうですね、ハハハ", "そうですね", 3, 0),
    )

    for text, words, sound_characters, chains in cases:
        split = split_vocal_sounds(text)
        assert (split.words, split.sound_characters, split.chains) == (
            words,
            sound_characters,
            chains,
        ), text
        assert normalize_words(text) == words, text


def test_duplicate_rule_marks_events_decoded_by_two_calls_not_one_split_call() -> None:
    event = _event(3, 0, 16_000, "猫が来た")
    first = _hypothesis(301, 0, 8_000, "猫が")
    second = _hypothesis(302, 8_000, 16_000, "来た")
    (component,) = build_temporal_components(
        (event,),
        (first, second),
        JA_TIME_DP_1_0,
    )
    alignment = align_component(component, JA_TIME_DP_1_0, normalize_minimal)
    assert alignment.event_alignments[0].hypothesis_ids == (
        first.hypothesis_id,
        second.hypothesis_id,
    )

    two_calls = duplicate_event_ids(
        alignment.event_alignments,
        {first.hypothesis_id: _id(401), second.hypothesis_id: _id(402)},
    )
    one_call = duplicate_event_ids(
        alignment.event_alignments,
        {first.hypothesis_id: _id(401), second.hypothesis_id: _id(401)},
    )

    assert two_calls == (event.event_id,)
    assert one_call == ()


def test_grouped_alignment_consumes_merge_and_fragments_once_in_time_order_with_event_ownership() -> (
    None
):
    first = _event(2, 0, 8_000, "東京")
    second = _event(3, 8_000, 16_000, "タワー")
    third = _event(4, 32_000, 48_000, "見える")
    merged = _hypothesis(102, 0, 16_000, "東京トすごいワー")
    fragment_one = _hypothesis(103, 32_000, 40_000, "見え")
    fragment_two = _hypothesis(104, 40_000, 48_000, "る")

    components = build_temporal_components(
        (first, second, third),
        (merged, fragment_one, fragment_two),
        JA_TIME_DP_1_0,
    )
    alignments = tuple(
        align_component(component, JA_TIME_DP_1_0, normalize_minimal)
        for component in components
    )

    assert len(alignments) == 2
    merge_alignment, fragment_alignment = alignments
    assert merge_alignment.status is AlignmentStatus.RESOLVED
    assert (
        merge_alignment.substitutions,
        merge_alignment.deletions,
        merge_alignment.insertions,
    ) == (1, 0, 3)
    assert len(merge_alignment.groups) == 1
    assert merge_alignment.groups[0].event_ids == (first.event_id, second.event_id)
    assert merge_alignment.groups[0].hypothesis_ids == (merged.hypothesis_id,)
    merge_by_event = {item.event_id: item for item in merge_alignment.event_alignments}
    assert (
        merge_by_event[first.event_id].substitutions,
        merge_by_event[first.event_id].deletions,
        merge_by_event[first.event_id].insertions,
    ) == (0, 0, 0)
    assert (
        merge_by_event[second.event_id].substitutions,
        merge_by_event[second.event_id].deletions,
        merge_by_event[second.event_id].insertions,
    ) == (1, 0, 3)

    assert fragment_alignment.status is AlignmentStatus.RESOLVED
    assert (
        fragment_alignment.substitutions,
        fragment_alignment.deletions,
        fragment_alignment.insertions,
    ) == (0, 0, 0)
    assert len(fragment_alignment.groups) == 1
    assert fragment_alignment.groups[0].event_ids == (third.event_id,)
    assert fragment_alignment.groups[0].hypothesis_ids == (
        fragment_one.hypothesis_id,
        fragment_two.hypothesis_id,
    )
    assert fragment_alignment.event_alignments[0].hypothesis_ids == (
        fragment_one.hypothesis_id,
        fragment_two.hypothesis_id,
    )

    consumed_hypotheses = tuple(
        hypothesis_id
        for alignment in alignments
        for group in alignment.groups
        for hypothesis_id in group.hypothesis_ids
    )
    assert consumed_hypotheses == (
        merged.hypothesis_id,
        fragment_one.hypothesis_id,
        fragment_two.hypothesis_id,
    )
    for alignment in alignments:
        assert (
            sum(item.substitutions for item in alignment.event_alignments),
            sum(item.deletions for item in alignment.event_alignments),
            sum(item.insertions for item in alignment.event_alignments),
        ) == (
            alignment.substitutions,
            alignment.deletions,
            alignment.insertions,
        )


def test_temporal_alignment_counts_orphans_and_duplicates_and_blocks_over_horizon_chains() -> (
    None
):
    missed = _event(5, 0, 8_000, "猫")
    hit = _event(6, 32_000, 40_000, "猫")
    chosen = _hypothesis(105, 32_000, 40_000, "猫")
    duplicate = _hypothesis(106, 32_000, 40_000, "猫")
    orphan = _hypothesis(107, 80_000, 88_000, "幻")

    local_components = build_temporal_components(
        (missed, hit),
        (chosen, duplicate, orphan),
        JA_TIME_DP_1_0,
    )
    local_alignments = tuple(
        align_component(component, JA_TIME_DP_1_0, normalize_minimal)
        for component in local_components
    )

    assert all(
        alignment.status is AlignmentStatus.RESOLVED for alignment in local_alignments
    )
    assert (
        sum(alignment.substitutions for alignment in local_alignments),
        sum(alignment.deletions for alignment in local_alignments),
        sum(alignment.insertions for alignment in local_alignments),
    ) == (0, 1, 2)
    local_by_event = {
        item.event_id: item
        for alignment in local_alignments
        for item in alignment.event_alignments
    }
    assert (
        local_by_event[missed.event_id].substitutions,
        local_by_event[missed.event_id].deletions,
        local_by_event[missed.event_id].insertions,
    ) == (0, 1, 0)
    assert (
        local_by_event[hit.event_id].substitutions,
        local_by_event[hit.event_id].deletions,
        local_by_event[hit.event_id].insertions,
    ) == (0, 0, 0)
    hit_alignment = next(
        alignment
        for alignment in local_alignments
        if any(item.event_id == hit.event_id for item in alignment.event_alignments)
    )
    assert hit_alignment.groups[0].hypothesis_ids == (chosen.hypothesis_id,)
    assert hit_alignment.unmatched_hypothesis_ids == (duplicate.hypothesis_id,)
    assert any(
        alignment.unmatched_event_ids == (missed.event_id,)
        for alignment in local_alignments
    )
    assert any(
        alignment.unmatched_hypothesis_ids == (orphan.hypothesis_id,)
        for alignment in local_alignments
    )

    # The whole chain would align perfectly, but it reaches 170,000 samples
    # beyond its longest member, past the 160,000-sample horizon.
    early = _event(7, 0, 8_000, "あい")
    middle = _event(8, 166_000, 174_000, "うえ")
    late = _event(9, 332_000, 340_000, "お")
    first_chain = _hypothesis(108, 0, 170_000, "あ")
    second_chain = _hypothesis(109, 170_000, 340_000, "いうえお")
    chain_components = build_temporal_components(
        (early, middle, late),
        (first_chain, second_chain),
        JA_TIME_DP_1_0,
    )

    assert len(chain_components) == 1
    chain_alignment = align_component(
        chain_components[0],
        JA_TIME_DP_1_0,
        normalize_minimal,
    )

    assert chain_alignment.status is AlignmentStatus.RESOLVED
    assert (
        chain_alignment.substitutions,
        chain_alignment.deletions,
        chain_alignment.insertions,
    ) == (0, 1, 1)
    assert tuple(
        (group.event_ids, group.hypothesis_ids) for group in chain_alignment.groups
    ) == (
        ((early.event_id,), (first_chain.hypothesis_id,)),
        ((middle.event_id, late.event_id), (second_chain.hypothesis_id,)),
    )


def test_alignment_matches_an_event_longer_than_the_horizon_to_its_identical_hypothesis() -> (
    None
):
    eleven_seconds = 176_000
    assert eleven_seconds > JA_TIME_DP_1_0.horizon_samples
    event = _event(10, 0, eleven_seconds, "今日はいい天気ですね")
    hypothesis = _hypothesis(110, 0, eleven_seconds, "今日はいい天気ですね")

    (alignment,) = _alignments((event,), (hypothesis,))

    assert alignment.status is AlignmentStatus.RESOLVED
    assert (
        alignment.substitutions,
        alignment.deletions,
        alignment.insertions,
    ) == (0, 0, 0)
    assert tuple(
        (group.event_ids, group.hypothesis_ids) for group in alignment.groups
    ) == (((event.event_id,), (hypothesis.hypothesis_id,)),)
    assert alignment.event_alignments[0].hypothesis_ids == (hypothesis.hypothesis_id,)


def test_first_observed_fate_table_keeps_output_event_local_and_honors_eof_precedence() -> (
    None
):
    arm_id = _id(500_001)
    names = (
        "invalid",
        "complete_miss",
        "partial_eof",
        "pending_eof",
        "vad_undetected",
        "zero_yield",
        "event_unrepresented",
        "wrapper_rejected",
        "wrapper_empty",
        "neighbor_preroll_success",
    )
    references: dict[str, ReferenceEvent] = {}
    scored: dict[str, ScoredEvent] = {}
    for index, name in enumerate(names, start=20):
        start = (index - 20) * 40_000
        reference = _reference_event(index, start, start + 8_000)
        references[name] = reference
        scored[name] = ScoredEvent(reference.event_id, reference.span, "猫")

    unrepresented_neighbor = _reference_event(30, 258_000, 266_000)
    preroll_trigger = _reference_event(31, 374_000, 382_000)
    references["event_unrepresented_neighbor"] = unrepresented_neighbor
    references["neighbor_preroll_trigger"] = preroll_trigger
    scored["event_unrepresented_neighbor"] = ScoredEvent(
        unrepresented_neighbor.event_id,
        unrepresented_neighbor.span,
        "別",
    )
    scored["neighbor_preroll_trigger"] = ScoredEvent(
        preroll_trigger.event_id,
        preroll_trigger.span,
        "隣",
    )

    records: list[EvidenceRecord] = []
    raw_hypotheses: list[ScoredHypothesis] = []

    def add_record(
        *,
        stage: Stage,
        kind: RecordKind,
        source_spans: tuple[SampleSpan, ...] = (),
        parent_ids: tuple[str, ...] = (),
        decision: str | None = None,
        diagnostics: tuple[DiagnosticCode, ...] = (),
        details: dict[str, object] | None = None,
    ) -> EvidenceRecord:
        record = _record(
            len(records) + 1,
            arm_id=arm_id,
            stage=stage,
            kind=kind,
            source_spans=source_spans,
            parent_ids=parent_ids,
            decision=decision,
            diagnostics=diagnostics,
            details=details,
        )
        records.append(record)
        return record

    complete_miss_diagnostic = add_record(
        stage=Stage.ENDPOINT,
        kind=RecordKind.DIAGNOSTIC,
        source_spans=(references["complete_miss"].span,),
        decision=DiagnosticCode.UNKNOWN_ENDPOINT_REASON.value,
    )
    partial_diagnostic = add_record(
        stage=Stage.ENDPOINT,
        kind=RecordKind.DIAGNOSTIC,
        source_spans=(references["partial_eof"].span,),
        decision=DiagnosticCode.ENDPOINT_PARTIAL_FRAME_EOF.value,
    )
    pending_diagnostic = add_record(
        stage=Stage.ENDPOINT,
        kind=RecordKind.DIAGNOSTIC,
        source_spans=(references["pending_eof"].span,),
        decision=DiagnosticCode.ENDPOINT_PENDING_EOF.value,
    )

    def add_decoder_path(
        name: str,
        *,
        raw_text: str | None,
        raw_span: SampleSpan | None = None,
        window_span: SampleSpan | None = None,
        wrapper_decision: str | None = None,
    ) -> tuple[EvidenceRecord, EvidenceRecord, EvidenceRecord | None]:
        event_span = references[name].span
        source_window = window_span or SampleSpan(
            event_span.timeline_id,
            event_span.start,
            max(event_span.end, raw_span.end if raw_span else event_span.end),
        )
        window = add_record(
            stage=Stage.ENDPOINT,
            kind=RecordKind.WINDOW,
            source_spans=(source_window,),
            decision=DiagnosticCode.UNKNOWN_ENDPOINT_REASON.value,
            details={"pcm_sha256": "a" * 64},
        )
        call = add_record(
            stage=Stage.DECODER,
            kind=RecordKind.DECODER_CALL,
            source_spans=window.source_spans,
            decision=(
                DiagnosticCode.DECODER_NO_YIELD.value
                if raw_text is None
                else None
            ),
            details={
                "call_mode": ArmKind.P0.value,
                "parent_call_record_id": None,
                "raw_segment_count": 0 if raw_text is None else 1,
                "source_artifact_id": window.artifact_id,
                "source_pcm_sha256": window.details["pcm_sha256"],
            },
        )
        if raw_text is None:
            return window, call, None
        source = event_span if raw_span is None else raw_span
        raw = add_record(
            stage=Stage.DECODER,
            kind=RecordKind.RAW_HYPOTHESIS,
            source_spans=(source,),
            parent_ids=(call.record_id,),
        )
        raw_hypotheses.append(
            ScoredHypothesis(raw.record_id, raw.source_spans, raw_text)
        )
        if wrapper_decision is not None:
            if wrapper_decision == DiagnosticCode.WRAPPER_CONFIDENCE_GATE.value:
                rejection = add_record(
                    stage=Stage.WRAPPER,
                    kind=RecordKind.WRAPPER_DECISION,
                    source_spans=raw.source_spans,
                    parent_ids=(raw.record_id,),
                    decision=wrapper_decision,
                )
                add_record(
                    stage=Stage.WRAPPER,
                    kind=RecordKind.WRAPPER_DECISION,
                    parent_ids=(rejection.record_id,),
                    decision=DiagnosticCode.WRAPPER_EMPTY.value,
                )
            else:
                add_record(
                    stage=Stage.WRAPPER,
                    kind=RecordKind.WRAPPER_DECISION,
                    source_spans=raw.source_spans,
                    parent_ids=(raw.record_id,),
                    decision=wrapper_decision,
                )
        return window, call, raw

    add_decoder_path("zero_yield", raw_text=None)
    unrepresented = references["event_unrepresented"].span
    _, _, unrepresented_raw = add_decoder_path(
        "event_unrepresented",
        raw_text="別",
        raw_span=unrepresented_neighbor.span,
        window_span=SampleSpan(
            unrepresented.timeline_id,
            unrepresented.start,
            unrepresented_neighbor.span.end,
        ),
        wrapper_decision=TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value,
    )
    add_decoder_path(
        "wrapper_rejected",
        raw_text="拒否",
        wrapper_decision=DiagnosticCode.WRAPPER_CONFIDENCE_GATE.value,
    )
    add_decoder_path(
        "wrapper_empty",
        raw_text="   ",
        wrapper_decision=DiagnosticCode.WRAPPER_EMPTY.value,
    )
    preroll_gate = add_record(
        stage=Stage.ENDPOINT,
        kind=RecordKind.GATE_TRANSITION,
        source_spans=(preroll_trigger.span,),
        details={"in_speech_before": False, "in_speech_after": True},
    )
    add_decoder_path(
        "neighbor_preroll_success",
        raw_text="成功",
        raw_span=references["neighbor_preroll_success"].span,
        window_span=SampleSpan(
            _TIMELINE_ID,
            references["neighbor_preroll_success"].span.start,
            preroll_trigger.span.end,
        ),
        wrapper_decision=TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value,
    )
    assert unrepresented_raw is not None
    assert not any(
        max(references["neighbor_preroll_success"].span.start, span.start)
        < min(references["neighbor_preroll_success"].span.end, span.end)
        for span in preroll_gate.source_spans
    )

    combined_names = tuple(name for name in names if name != "vad_undetected")
    combined_reference_names = (
        *combined_names,
        "event_unrepresented_neighbor",
        "neighbor_preroll_trigger",
    )
    ordered_references = tuple(
        sorted(
            (references[name] for name in combined_reference_names),
            key=lambda event: (event.span.start, event.span.end, event.event_id),
        )
    )
    scored_by_id = {event.event_id: event for event in scored.values()}
    ordered_scored = tuple(scored_by_id[event.event_id] for event in ordered_references)
    arm = ArmEvidence(
        arm_id=arm_id,
        kind=ArmKind.P0,
        status=ArmStatus.COMPLETE,
        records=tuple(records),
        scored_events=ordered_scored,
        raw_hypotheses=tuple(raw_hypotheses),
        reference_invalid_event_ids=(references["invalid"].event_id,),
        alignment_policy=JA_TIME_DP_1_0,
    )
    sink = _RecordingFateSink()

    combined_fates = derive_event_fates(
        ordered_references,
        arm,
        _alignments(ordered_scored, tuple(raw_hypotheses)),
        sink,
    )
    vad_arm_id = _id(500_011)
    vad_diagnostic = _record(
        201,
        arm_id=vad_arm_id,
        stage=Stage.ENDPOINT,
        kind=RecordKind.DIAGNOSTIC,
        decision=DiagnosticCode.VAD_NOT_DETECTED.value,
    )
    vad_scored = (scored["vad_undetected"],)
    vad_arm = ArmEvidence(
        arm_id=vad_arm_id,
        kind=ArmKind.P0,
        status=ArmStatus.COMPLETE,
        records=(vad_diagnostic,),
        scored_events=vad_scored,
        raw_hypotheses=(),
        reference_invalid_event_ids=(),
        alignment_policy=JA_TIME_DP_1_0,
    )
    vad_fates = derive_event_fates(
        (references["vad_undetected"],),
        vad_arm,
        _alignments(vad_scored, ()),
        sink,
    )
    fates = (*combined_fates, *vad_fates)

    by_name = {
        name: next(fate for fate in fates if fate.event_id == references[name].event_id)
        for name in names
    }
    expected = {
        "invalid": (
            TerminalOutcome.REFERENCE_INVALID,
            Stage.REFERENCE,
            TerminalOutcome.REFERENCE_INVALID.value,
        ),
        "complete_miss": (
            TerminalOutcome.NO_DECODER_WINDOW,
            Stage.ENDPOINT,
            DiagnosticCode.UNKNOWN_ENDPOINT_REASON.value,
        ),
        "partial_eof": (
            TerminalOutcome.NO_DECODER_WINDOW,
            Stage.VAD,
            DiagnosticCode.ENDPOINT_PARTIAL_FRAME_EOF.value,
        ),
        "pending_eof": (
            TerminalOutcome.NO_DECODER_WINDOW,
            Stage.ENDPOINT,
            DiagnosticCode.ENDPOINT_PENDING_EOF.value,
        ),
        "vad_undetected": (
            TerminalOutcome.NO_DECODER_WINDOW,
            Stage.VAD,
            DiagnosticCode.VAD_NOT_DETECTED.value,
        ),
        "zero_yield": (
            TerminalOutcome.DECODER_NO_YIELD,
            Stage.DECODER,
            TerminalOutcome.DECODER_NO_YIELD.value,
        ),
        "event_unrepresented": (
            TerminalOutcome.DECODER_EVENT_UNREPRESENTED,
            Stage.DECODER,
            TerminalOutcome.DECODER_EVENT_UNREPRESENTED.value,
        ),
        "wrapper_rejected": (
            TerminalOutcome.WRAPPER_REJECTED,
            Stage.WRAPPER,
            TerminalOutcome.WRAPPER_REJECTED.value,
        ),
        "wrapper_empty": (
            TerminalOutcome.WRAPPER_EMPTY,
            Stage.WRAPPER,
            TerminalOutcome.WRAPPER_EMPTY.value,
        ),
        "neighbor_preroll_success": (
            TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE,
            None,
            TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value,
        ),
    }
    assert {
        name: (fate.outcome, fate.first_missing_stage, fate.reason)
        for name, fate in by_name.items()
    } == expected
    assert complete_miss_diagnostic.record_id in by_name[
        "complete_miss"
    ].diagnostic_record_ids
    assert partial_diagnostic.record_id in by_name["partial_eof"].diagnostic_record_ids
    assert vad_diagnostic.record_id not in by_name["partial_eof"].diagnostic_record_ids
    assert pending_diagnostic.record_id in by_name["pending_eof"].diagnostic_record_ids
    assert unrepresented_raw.record_id in by_name[
        "event_unrepresented"
    ].evidence_record_ids
    rejection_gate = next(
        record
        for record in records
        if record.kind is RecordKind.WRAPPER_DECISION
        and record.decision == DiagnosticCode.WRAPPER_CONFIDENCE_GATE.value
    )
    assert rejection_gate.record_id in by_name["wrapper_rejected"].diagnostic_record_ids
    assert next(
        fate
        for fate in combined_fates
        if fate.event_id == unrepresented_neighbor.event_id
    ).outcome is TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE
    assert len({(fate.event_id, fate.arm_id) for fate in fates}) == len(fates)
    assert sum(record.kind is RecordKind.EVENT_FATE for record in sink.records) == len(
        fates
    )


def _window_covering_one_and_a_half_events() -> SimpleNamespace:
    """Two adjacent events; one decoder window covers the first and half the second."""

    arm_id = _id(500_002)
    first_reference = _reference_event(40, 0, 8_000)
    second_reference = _reference_event(41, 8_000, 16_000)
    first = ScoredEvent(first_reference.event_id, first_reference.span, "猫")
    second = ScoredEvent(second_reference.event_id, second_reference.span, "犬")
    window = _record(
        101,
        arm_id=arm_id,
        stage=Stage.ENDPOINT,
        kind=RecordKind.WINDOW,
        source_spans=(SampleSpan(_TIMELINE_ID, 0, 12_000),),
        decision=DiagnosticCode.UNKNOWN_ENDPOINT_REASON.value,
        details={"pcm_sha256": "b" * 64},
    )
    call = _record(
        102,
        arm_id=arm_id,
        stage=Stage.DECODER,
        kind=RecordKind.DECODER_CALL,
        source_spans=window.source_spans,
        details={
            "call_mode": ArmKind.P0.value,
            "parent_call_record_id": None,
            "raw_segment_count": 1,
            "source_artifact_id": window.artifact_id,
            "source_pcm_sha256": window.details["pcm_sha256"],
        },
    )
    raw_record = _record(
        103,
        arm_id=arm_id,
        stage=Stage.DECODER,
        kind=RecordKind.RAW_HYPOTHESIS,
        source_spans=window.source_spans,
        parent_ids=(call.record_id,),
    )
    wrapper = _record(
        104,
        arm_id=arm_id,
        stage=Stage.WRAPPER,
        kind=RecordKind.WRAPPER_DECISION,
        source_spans=window.source_spans,
        parent_ids=(raw_record.record_id,),
        decision=TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value,
    )
    raw = ScoredHypothesis(raw_record.record_id, raw_record.source_spans, "猫犬")
    arm = ArmEvidence(
        arm_id=arm_id,
        kind=ArmKind.P0,
        status=ArmStatus.COMPLETE,
        records=(window, call, raw_record, wrapper),
        scored_events=(first, second),
        raw_hypotheses=(raw,),
        reference_invalid_event_ids=(),
        alignment_policy=JA_TIME_DP_1_0,
    )
    return SimpleNamespace(
        references=(first_reference, second_reference),
        arm=arm,
        alignments=_alignments((first, second), (raw,)),
    )


def test_event_fates_are_unique_and_usability_enforces_short_and_long_boundaries() -> (
    None
):
    case = _window_covering_one_and_a_half_events()
    sink = _RecordingFateSink()

    fates = derive_event_fates(case.references, case.arm, case.alignments, sink)

    assert len(fates) == 2
    assert len({(fate.event_id, fate.arm_id) for fate in fates}) == 2
    assert all(
        fate.outcome is TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE
        and fate.first_missing_stage is None
        for fate in fates
    )
    assert sum(record.kind is RecordKind.EVENT_FATE for record in sink.records) == 2
    assert any(
        record.kind is RecordKind.DIAGNOSTIC
        and record.decision == DiagnosticCode.ENDPOINT_BOUNDARY_TRUNCATED.value
        for record in sink.records
    )

    cases = (
        (50, "猫犬鳥魚", "猫犬鳥魚", True),
        (51, "猫犬鳥魚", "猫犬鳥馬", False),
        (52, "一二三四五六七八", "一二三四五六九十", True),
        (53, "一二三四五六七八", "一二三四五九十百", False),
        (54, "", "", False),
    )
    for ordinal, reference_text, hypothesis_text, expected in cases:
        reference = _reference_event(ordinal, ordinal * 20_000, ordinal * 20_000 + 8_000)
        alignment = _event_alignment(
            reference.event_id,
            _id(400_000 + ordinal),
            reference_text,
            hypothesis_text,
        )
        assert (
            event_is_lexically_usable(
                reference,
                TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE,
                alignment,
                JA_EVENT_USABLE_1_1,
            )
            is expected
        )
    assert (
        event_is_lexically_usable(
            _reference_event(55, 1_200_000, 1_208_000),
            TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE,
            None,
            JA_EVENT_USABLE_1_1,
        )
        is None
    )


class _ParentCheckingLedger(_RecordingFateSink):
    """Assigns its own record IDs, as LedgerWriter does, and refuses unwritten parents."""

    def __init__(self, written_ids: set[str]) -> None:
        super().__init__()
        self.written_ids = set(written_ids)

    def emit(self, draft: EvidenceRecordDraft) -> EvidenceRecord:
        assert set(draft.parent_ids) <= self.written_ids, "parent was never written"
        record = super().emit(draft)
        self.written_ids.add(record.record_id)
        return record


def test_run_fate_buffer_links_the_fate_to_the_boundary_diagnostic_it_wrote() -> None:
    case = _window_covering_one_and_a_half_events()
    buffer = runtime._FateDraftBuffer()

    derive_event_fates(case.references, case.arm, case.alignments, buffer)
    ledger = _ParentCheckingLedger({record.record_id for record in case.arm.records})
    buffer.flush(ledger)

    (diagnostic,) = (r for r in ledger.records if r.kind is RecordKind.DIAGNOSTIC)
    truncated_fate = next(
        r
        for r in ledger.records
        if r.kind is RecordKind.EVENT_FATE
        and r.artifact_id == case.references[1].event_id
    )
    assert diagnostic.decision == DiagnosticCode.ENDPOINT_BOUNDARY_TRUNCATED.value
    assert diagnostic.record_id in truncated_fate.parent_ids


def test_fate_does_not_credit_a_dropped_event_with_a_neighbor_hypothesis_inside_the_collar() -> (
    None
):
    arm_id = _id(500_004)
    spoken_reference = _reference_event(42, 0, 16_000)
    dropped_reference = _reference_event(43, 18_000, 26_000)
    spoken = ScoredEvent(spoken_reference.event_id, spoken_reference.span, "猫")
    dropped = ScoredEvent(dropped_reference.event_id, dropped_reference.span, "犬")
    window = _record(
        111,
        arm_id=arm_id,
        stage=Stage.ENDPOINT,
        kind=RecordKind.WINDOW,
        source_spans=(SampleSpan(_TIMELINE_ID, 0, 26_000),),
        decision=DiagnosticCode.UNKNOWN_ENDPOINT_REASON.value,
        details={"pcm_sha256": "c" * 64},
    )
    call = _record(
        112,
        arm_id=arm_id,
        stage=Stage.DECODER,
        kind=RecordKind.DECODER_CALL,
        source_spans=window.source_spans,
        details={
            "call_mode": ArmKind.P0.value,
            "parent_call_record_id": None,
            "raw_segment_count": 1,
            "source_artifact_id": window.artifact_id,
            "source_pcm_sha256": window.details["pcm_sha256"],
        },
    )
    # The decoder emits only the first line. Its segment ends 125 ms before the
    # dropped line starts, inside the 250 ms collar.
    raw_record = _record(
        113,
        arm_id=arm_id,
        stage=Stage.DECODER,
        kind=RecordKind.RAW_HYPOTHESIS,
        source_spans=(spoken_reference.span,),
        parent_ids=(call.record_id,),
    )
    wrapper = _record(
        114,
        arm_id=arm_id,
        stage=Stage.WRAPPER,
        kind=RecordKind.WRAPPER_DECISION,
        source_spans=raw_record.source_spans,
        parent_ids=(raw_record.record_id,),
        decision=TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value,
    )
    raw = ScoredHypothesis(raw_record.record_id, raw_record.source_spans, "猫")
    arm = ArmEvidence(
        arm_id=arm_id,
        kind=ArmKind.P0,
        status=ArmStatus.COMPLETE,
        records=(window, call, raw_record, wrapper),
        scored_events=(spoken, dropped),
        raw_hypotheses=(raw,),
        reference_invalid_event_ids=(),
        alignment_policy=JA_TIME_DP_1_0,
    )

    fates = derive_event_fates(
        (spoken_reference, dropped_reference),
        arm,
        _alignments((spoken, dropped), (raw,)),
        _RecordingFateSink(),
    )

    by_event = {fate.event_id: fate for fate in fates}
    assert (
        by_event[spoken.event_id].outcome
        is TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE
    )
    assert (
        by_event[dropped.event_id].outcome,
        by_event[dropped.event_id].first_missing_stage,
    ) == (TerminalOutcome.DECODER_EVENT_UNREPRESENTED, Stage.DECODER)
    assert wrapper.record_id not in by_event[dropped.event_id].evidence_record_ids


def test_metrics_keep_misses_orphans_unknowns_and_aggregate_by_title() -> None:
    arm_id = _id(500_003)
    title_a_references = (
        _reference_event(60, 0, 8_000),
        _reference_event(61, 24_000, 32_000),
        _reference_event(62, 48_000, 56_000),
    )
    title_a_events = tuple(
        ScoredEvent(reference.event_id, reference.span, text)
        for reference, text in zip(title_a_references, ("猫", "犬", "鳥"), strict=True)
    )
    title_a_hypotheses = tuple(
        ScoredHypothesis(
            _id(400_000 + index),
            (event.span,),
            event.text or "",
        )
        for index, event in enumerate(title_a_events, start=60)
    )
    title_b_reference = _reference_event(63, 0, 8_000)
    title_b_event = ScoredEvent(
        title_b_reference.event_id,
        title_b_reference.span,
        "猫犬",
    )
    orphan = ScoredHypothesis(
        _id(400_163),
        (SampleSpan(_TIMELINE_ID, 100_000, 108_000),),
        "幻",
    )

    def fate(reference: ReferenceEvent, outcome: TerminalOutcome) -> EventFate:
        if outcome is TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE:
            stage = None
        elif outcome is TerminalOutcome.UNKNOWN:
            stage = Stage.RUN
        else:
            stage = Stage.VAD
        return EventFate(
            event_id=reference.event_id,
            arm_id=arm_id,
            outcome=outcome,
            first_missing_stage=stage,
            reason=outcome.value,
            evidence_record_ids=(),
            diagnostic_record_ids=(),
        )

    observed_latency = ScalarObservation(
        status=ObservationStatus.OBSERVED,
        value=0.25,
        unit="seconds",
    )
    missing_latency = ScalarObservation(
        status=ObservationStatus.INSTRUMENTATION_MISSING,
        value=None,
        unit="seconds",
    )
    out_of_scope_resources = ResourceObservations(
        rtf=ScalarObservation(ObservationStatus.OUT_OF_SCOPE, None, "ratio"),
        cold_start_seconds=ScalarObservation(
            ObservationStatus.OUT_OF_SCOPE,
            None,
            "seconds",
        ),
        peak_rss_bytes=ScalarObservation(
            ObservationStatus.OUT_OF_SCOPE,
            None,
            "bytes",
        ),
        peak_vram_bytes=ScalarObservation(
            ObservationStatus.OUT_OF_SCOPE,
            None,
            "bytes",
        ),
    )
    title_a_scores = (
        EventScore(
            title_a_references[0],
            fate(
                title_a_references[0],
                TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE,
            ),
            raw_nonblank=True,
            lexically_usable=True,
            paced_deadline_met=True,
            output_latency_seconds=observed_latency,
        ),
        EventScore(
            title_a_references[1],
            fate(
                title_a_references[1],
                TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE,
            ),
            raw_nonblank=True,
            lexically_usable=True,
            paced_deadline_met=True,
            output_latency_seconds=observed_latency,
        ),
        EventScore(
            title_a_references[2],
            fate(title_a_references[2], TerminalOutcome.UNKNOWN),
            raw_nonblank=None,
            lexically_usable=None,
            paced_deadline_met=None,
            output_latency_seconds=missing_latency,
        ),
    )
    title_b_scores = (
        EventScore(
            title_b_reference,
            fate(title_b_reference, TerminalOutcome.NO_DECODER_WINDOW),
            raw_nonblank=False,
            lexically_usable=False,
            paced_deadline_met=False,
            output_latency_seconds=missing_latency,
        ),
    )
    title_a = ScoredFixture(
        fixture=FixtureManifest(
            fixture_id=_id(510_001),
            partition=Partition.LOCKED_CONFIRMATION,
            title_id=_id(520_001),
            rights_record_id=_id(530_001),
        ),
        arm_id=arm_id,
        timeline_id=_TIMELINE_ID,
        sample_rate_hz=16_000,
        sample_count=80_000,
        model_runtime_sha256="a" * 64,
        events=title_a_scores,
        normalized_alignments=tuple(
            _normalization_set(level, title_a_events, title_a_hypotheses)
            for level in NormalizationLevel
        ),
        expected_wrapper_hypothesis_ids=tuple(
            hypothesis.hypothesis_id for hypothesis in title_a_hypotheses
        ),
        vad_observation_status=ObservationStatus.OUT_OF_SCOPE,
        vad_positive_spans=(),
        decoder_window_observation_status=ObservationStatus.OUT_OF_SCOPE,
        decoder_window_spans=(),
        boundary_truncated_event_ids=(),
        duplicate_event_ids=(),
        duplicate_observation_status=ObservationStatus.OBSERVED,
        hard_event_ids=(),
        hard_stratum_version="test-hard-stratum-1.0",
        hard_stratum_sha256="b" * 64,
        non_speech_control_spans=(),
        hallucination_output_ids=(),
        resources=out_of_scope_resources,
        vocal_sounds=_NO_VOCAL_SOUNDS,
    )
    title_b = ScoredFixture(
        fixture=FixtureManifest(
            fixture_id=_id(510_002),
            partition=Partition.LOCKED_CONFIRMATION,
            title_id=_id(520_002),
            rights_record_id=_id(530_002),
        ),
        arm_id=arm_id,
        timeline_id=_TIMELINE_ID,
        sample_rate_hz=16_000,
        sample_count=1_040_000,
        model_runtime_sha256="a" * 64,
        events=title_b_scores,
        normalized_alignments=tuple(
            _normalization_set(level, (title_b_event,), (orphan,))
            for level in NormalizationLevel
        ),
        expected_wrapper_hypothesis_ids=(orphan.hypothesis_id,),
        vad_observation_status=ObservationStatus.OUT_OF_SCOPE,
        vad_positive_spans=(),
        decoder_window_observation_status=ObservationStatus.OUT_OF_SCOPE,
        decoder_window_spans=(),
        boundary_truncated_event_ids=(),
        duplicate_event_ids=(),
        duplicate_observation_status=ObservationStatus.OBSERVED,
        hard_event_ids=(),
        hard_stratum_version="test-hard-stratum-1.0",
        hard_stratum_sha256="b" * 64,
        non_speech_control_spans=(
            SampleSpan(_TIMELINE_ID, 80_000, 1_040_000),
        ),
        hallucination_output_ids=(orphan.hypothesis_id,),
        resources=out_of_scope_resources,
        vocal_sounds=_NO_VOCAL_SOUNDS,
    )

    report = compute_metrics((title_a, title_b), seed=105, bootstrap_replicates=256)
    repeated = compute_metrics(
        (title_a, title_b),
        seed=105,
        bootstrap_replicates=256,
    )

    cer = report.micro.cer_minimal
    assert (
        cer.substitutions,
        cer.deletions,
        cer.insertions,
        cer.reference_characters,
    ) == (0, 2, 1, 5)
    assert cer.status is MetricStatus.DEFINED
    assert cer.value == 3 / 5
    assert (
        report.micro.wrapper_available_recall.numerator,
        report.micro.wrapper_available_recall.denominator,
        report.micro.wrapper_available_recall.unknown,
    ) == (2, 4, 1)
    assert report.micro.wrapper_available_recall.value == 1 / 2
    assert math.isclose(report.title_macro.wrapper_available_recall.value or 0.0, 1 / 3)
    assert report.unknown_counts.event_fates == 1
    assert report.micro.hallucinations.per_minute == 1.0
    assert (
        report.micro.deadline_recall.numerator,
        report.micro.deadline_recall.denominator,
    ) == (2, 4)
    assert report.bootstrap_intervals
    assert report.bootstrap_intervals == repeated.bootstrap_intervals
    wrapper_interval = next(
        interval
        for interval in report.bootstrap_intervals
        if interval.metric_name == "wrapper_available_recall"
    )
    minimal_cer_interval = next(
        interval
        for interval in report.bootstrap_intervals
        if interval.metric_name == "cer_minimal"
    )
    assert wrapper_interval.status is MetricStatus.INCOMPLETE
    assert (wrapper_interval.lower, wrapper_interval.upper) == (None, None)
    assert (minimal_cer_interval.lower, minimal_cer_interval.upper) == (0.0, 1.5)


def _out_of_scope_resources() -> ResourceObservations:
    return ResourceObservations(
        rtf=ScalarObservation(ObservationStatus.OUT_OF_SCOPE, None, "ratio"),
        cold_start_seconds=ScalarObservation(
            ObservationStatus.OUT_OF_SCOPE, None, "seconds"
        ),
        peak_rss_bytes=ScalarObservation(ObservationStatus.OUT_OF_SCOPE, None, "bytes"),
        peak_vram_bytes=ScalarObservation(
            ObservationStatus.OUT_OF_SCOPE, None, "bytes"
        ),
    )


def _plain_fate(
    reference: ReferenceEvent,
    arm_id: str,
    outcome: TerminalOutcome,
) -> EventFate:
    return EventFate(
        event_id=reference.event_id,
        arm_id=arm_id,
        outcome=outcome,
        first_missing_stage=(
            None if outcome is TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE else Stage.VAD
        ),
        reason=outcome.value,
        evidence_record_ids=(),
        diagnostic_record_ids=(),
    )


def _plain_fixture(
    ordinal: int,
    arm_id: str,
    sample_count: int,
    events: tuple[EventScore, ...],
    scored_events: tuple[ScoredEvent, ...],
    hypotheses: tuple[ScoredHypothesis, ...],
) -> ScoredFixture:
    return ScoredFixture(
        fixture=FixtureManifest(
            fixture_id=_id(510_000 + ordinal),
            partition=Partition.LOCKED_CONFIRMATION,
            title_id=_id(520_000 + ordinal),
            rights_record_id=_id(530_000 + ordinal),
        ),
        arm_id=arm_id,
        timeline_id=_TIMELINE_ID,
        sample_rate_hz=16_000,
        sample_count=sample_count,
        model_runtime_sha256="a" * 64,
        events=events,
        normalized_alignments=tuple(
            _normalization_set(level, scored_events, hypotheses)
            for level in NormalizationLevel
        ),
        expected_wrapper_hypothesis_ids=tuple(
            hypothesis.hypothesis_id for hypothesis in hypotheses
        ),
        vad_observation_status=ObservationStatus.OUT_OF_SCOPE,
        vad_positive_spans=(),
        decoder_window_observation_status=ObservationStatus.OUT_OF_SCOPE,
        decoder_window_spans=(),
        boundary_truncated_event_ids=(),
        duplicate_event_ids=(),
        duplicate_observation_status=ObservationStatus.OBSERVED,
        hard_event_ids=(),
        hard_stratum_version="test-hard-stratum-1.0",
        hard_stratum_sha256="b" * 64,
        non_speech_control_spans=(),
        hallucination_output_ids=(),
        resources=_out_of_scope_resources(),
        vocal_sounds=_NO_VOCAL_SOUNDS,
    )


def test_unintelligible_neighbor_bridged_by_one_hypothesis_keeps_the_line_lexically_scored() -> (
    None
):
    arm_id = _id(500_005)
    line = _reference_event(70, 0, 16_000)
    # The mumble starts 1,000 samples (about 62 ms) after the line ends.
    mumble = ReferenceEvent(
        event_id=_id(71),
        master_id=_id(900_001),
        span=SampleSpan(_TIMELINE_ID, 17_000, 25_000),
        intelligibility="unintelligible",
        verbatim_ref_id=None,
        tags=(),
        annotation_version="test-reference-1.0",
    )
    assert line.verbatim_ref_id is not None
    line_text_id = _id(610_070)
    reference_set = SimpleNamespace(
        events=(line, mumble),
        verbatim_references=(
            VerbatimReference(
                verbatim_ref_id=line.verbatim_ref_id,
                event_ids=(line.event_id,),
                text_id=line_text_id,
                annotation_version="test-reference-1.0",
            ),
        ),
    )
    scored = scored_reference_events(reference_set, {line_text_id: "今日は"})
    bridge = _hypothesis(170, 0, 25_000, "今日は")
    # A stray syllable far from the mumble stays an ordinary insertion; it must
    # be counted rather than left unassigned because the component also holds
    # an event without lexical text.
    stray = _hypothesis(171, 0, 4_000, "あ")
    hypotheses = (stray, bridge)
    minimal = _normalization_set(NormalizationLevel.MINIMAL, scored, hypotheses)
    line_alignment = next(
        (
            item
            for component in minimal.components
            for item in component.event_alignments
            if item.event_id == line.event_id
        ),
        None,
    )
    line_usable = event_is_lexically_usable(
        line,
        TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE,
        line_alignment,
        JA_EVENT_USABLE_1_1,
    )
    latency = ScalarObservation(ObservationStatus.OBSERVED, 0.25, "seconds")
    fixture = _plain_fixture(
        70,
        arm_id,
        32_000,
        (
            EventScore(
                line,
                _plain_fate(line, arm_id, TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE),
                raw_nonblank=True,
                lexically_usable=line_usable,
                paced_deadline_met=True,
                output_latency_seconds=latency,
            ),
            EventScore(
                mumble,
                _plain_fate(
                    mumble, arm_id, TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE
                ),
                raw_nonblank=None,
                lexically_usable=None,
                paced_deadline_met=None,
                output_latency_seconds=latency,
            ),
        ),
        scored,
        hypotheses,
    )

    report = compute_metrics((fixture,), bootstrap_replicates=0)

    assert line_usable is True
    cer = report.micro.cer_minimal
    assert (
        cer.status,
        cer.reference_characters,
        cer.insertions,
        cer.unassigned_insertions,
    ) == (MetricStatus.DEFINED, 3, 1, 0)
    assert cer.value == 1 / 3
    assert (
        report.micro.lexically_usable_recall.numerator,
        report.micro.lexically_usable_recall.denominator,
        report.micro.lexically_usable_recall.unknown,
    ) == (1, 1, 0)


def test_title_bootstrap_lower_bound_is_floored_by_the_exact_pooled_bound() -> None:
    arm_id = _id(500_006)
    fixtures: list[ScoredFixture] = []
    for title_index, available_count in enumerate((39, 40, 38)):
        references = tuple(
            _reference_event(
                1_000 * (title_index + 1) + index,
                index * 16_000,
                index * 16_000 + 8_000,
            )
            for index in range(40)
        )
        scored = tuple(
            ScoredEvent(reference.event_id, reference.span, "猫")
            for reference in references
        )
        latency = ScalarObservation(ObservationStatus.OBSERVED, 0.25, "seconds")
        events = tuple(
            EventScore(
                reference,
                _plain_fate(
                    reference,
                    arm_id,
                    TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE
                    if index < available_count
                    else TerminalOutcome.NO_DECODER_WINDOW,
                ),
                raw_nonblank=index < available_count,
                lexically_usable=False,
                paced_deadline_met=False,
                output_latency_seconds=latency,
            )
            for index, reference in enumerate(references)
        )
        fixtures.append(
            _plain_fixture(80 + title_index, arm_id, 640_000, events, scored, ())
        )

    report = compute_metrics(tuple(fixtures), seed=105, bootstrap_replicates=10_000)

    recall = report.micro.wrapper_available_recall
    assert (recall.numerator, recall.denominator) == (117, 120)
    interval = next(
        item
        for item in report.bootstrap_intervals
        if item.metric_name == "wrapper_available_recall"
    )
    assert interval.status is MetricStatus.DEFINED
    assert interval.title_count == 3
    assert interval.lower is not None and interval.lower < 0.95
    # scipy.stats.beta.ppf(0.025, 117, 4): the exact one-sided pooled bound.
    assert math.isclose(interval.lower, 0.928676805843095, abs_tol=1e-9)
