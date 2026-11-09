"""Unit tests for the polarity detector (yamabiko/translation/negation.py).

The Iron Law drill for each: a one-line production defect that reddens it.
  - drop 終止形/連用形 restriction -> obligation idioms (行かねば) start firing
    -> the "does not fire" cases go red.
  - remove the rhetorical exclusion -> いいじゃないか fires -> red.
  - remove the cluster walk -> 食べない人が来た fires -> red.
  - swap negated's lemma set -> the sentence-final negation cases go red.
  - drop the grammaticalized-idiom exclusion (_is_affirmative_idiom) ->
    かもしれない / なければいけない / 行くしかない / 間違いない / しようがない /
    たまらない start firing -> the grammaticalized-affirmative cases go red.
  - anchor _is_affirmative_idiom on pred_tokens[-1] instead of walking past the
    trailing copula/aux (_final_neg_anchor) -> the polite (…いけません) and
    copula-suffixed (…ないだろう / …ないでしょう / …ないです) idioms start firing
    -> the polite/formal grammaticalized-affirmative cases go red.
  - over-widen that exclusion (defuse standalone いけない, or any N+が+ない) ->
    それはいけない / 元気がない stop firing -> the honesty-guard cases go red.
  - widen (H) to a bare 得る match (drop the ざる+を prefix checks) ->
    あり得ない / あり得ません stop firing -> those honesty guards go red.
  - drop the 接尾辞 arm from (G)'s gate -> なりかねない (tagged 接尾辞/動詞的 by
    unidic-lite, the bundled dictionary) fires -> the (G) cases go red.
  - require the interrogative か to sit DIRECTLY on the negation marker again ->
    行きませんかね / 早く来ないかな / 飲みに行かないですか false-fire -> the
    interrogative cases go red.
  - widen the interrogative gate's marker walk to skip 準体助詞 as well as
    助動詞 -> 行かないのか / 行かないんですか stop firing -> the genuine
    negative-question fixtures go red.
  - drop the せて/させて requirement from causative_grant -> bare-te requests
    (静かにして) and plain てください start firing -> the "not a grant" cases go red.
  - drop the grant-tail ($-anchored くれ/ください/やれ/やって) requirement ->
    declarative 見せてくれる / 見せてくれた start firing -> red.
  - remove cannot/unable/impossible from _EN_NEG -> those go red.
"""
from __future__ import annotations

import pytest

from yamabiko.translation.negation import (
    JpPolarity,
    analyze_jp,
    en_has_added_imperative_negation,
    en_has_negation,
)


# --------------------------------------------------------------------------
# JP negation detection (direction A)
# --------------------------------------------------------------------------

# Real fixtures from the 2026-07-05 judged eval plus the core inflection
# families. Each is a CLEAR sentence-final negation that must fire.
_NEGATED = [
    "応用できてない",     # eval fixture: negation dropped in EN
    "できてない",         # bare -te-nai
    "わかりません",       # -masen (ん lemma=ず)
    "行かなかった",       # -nakatta
    "知らない",           # plain -nai
    "行けない",           # potential -nai (行ける -> 行けない) -- F2 premise
    "行かないで",         # -naide request-negative
    "やめておらず",       # -orazu literary
    "わからん",           # colloquial -n (=ず)
    "元気がない",         # existential 形容詞 negation
    "ちゃんと応用できてないな",  # -nai + final particle な (must strip な, still fire)
    "食べない",           # plain -nai baseline
    "分からない",         # plain -nai
    "知りません",         # -masen
    "言えない",           # potential -nai (言える -> 言えない)
    # Honesty guards for the grammaticalized-idiom exclusion below: these carry
    # a GENUINE negation and MUST keep firing (the exclusion is narrow).
    "それはいけない",     # standalone いけない (no conditional chain) = negative
    "食べてはいけない",   # 〜てはいけない = "must not" (single negative) = negative
    "それはいけません",   # polite honesty guard: no earlier neg -> still fires
    "食べてはいけません", # polite prohibition "must not eat" -> still fires
    # Real negations carrying a trailing copula/aux (だろう/でしょう/です) -- the
    # aux must not defuse a genuine negation (the idiom-anchor skip is stem-gated).
    "食べないだろう",     # "probably won't eat"
    "元気がないでしょう", # existential negation (元気 is not a whitelisted stem)
    "知らないです",       # plain -nai + polite です
    # Honesty guards for the POLITE ある/ます-negative idiom exclusion: standalone
    # ありません IS a real existential negation ("there is no X"), and a plain
    # polite …ません verb negation must keep firing. The idiom re-point is
    # anchor-gated (しか/違い/間違い/しようが/しかたが/堪る only), so these do NOT
    # match the whitelist and stay negated.
    "お金がありません",   # existential ありません (金 is not a whitelisted stem)
    "時間がありません",   # existential ありません (時間 is not a whitelisted stem)
    "行きません",         # plain polite verb negation (行き ませ ん)
    # Honesty guards for the interrogative-か gate: the 準体助詞 の (surfacing
    # as の or ん) nominalizes the negation before the か, marking a GENUINE
    # negative question ("is it that you're not going?"), so these must keep
    # firing. The gate's marker walk stops at the 準体助詞; widening it to any
    # trailing か would silently defuse them.
    "行かないのか",       # genuine negative question (ない + の + か)
    "行かないんですか",   # colloquial negative question (ない + ん(=の) + です + か)
    # Honesty guards for the ざるを得ない exclusion (H): あり得ない is a real
    # potential negation ("impossible", common anime dialogue), tokenized
    # あり + 得(得る) + ない. (H) is gated on the ざる+を prefix before 得る;
    # a bare 得る+ない match would silently defuse these.
    "あり得ない",         # potential negation "impossible" (得る + ない)
    "あり得ません",       # polite potential negation (得 + ませ + ん)
]

# Must NOT be read as a final-predicate negation.
_NOT_NEGATED = [
    "今日はいい天気です",   # plain affirmative declarative
    "食べない人が来た",     # ない is a relative-clause modifier; 来た is affirmative
    "いいじゃないか",       # rhetorical affirmation (brief-mandated exclusion)
    "これは本じゃない",     # copular ...じゃない (conservative exclusion)
    "いいじゃん",           # rhetorical じゃん
    "行かなきゃ",           # obligation idiom "gotta go" (仮定形)
    "行かなければ",         # obligation idiom (仮定形 + ば)
    "行かないと",           # obligation idiom (と breaks the cluster)
    "行かねば",             # obligation idiom (仮定形)
    "死なせてやってくれ",   # eval fixture: a REQUEST, not a negation
    "行くな",               # prohibitive, but carries no negation aux
    "待って",               # te-form request
]


@pytest.mark.parametrize("text", _NEGATED)
def test_analyze_jp_marks_sentence_final_negation(text, tagger):
    assert analyze_jp(text, tagger).negated is True, (
        f"{text!r} should be detected as a final-predicate negation"
    )


@pytest.mark.parametrize("text", _NOT_NEGATED)
def test_analyze_jp_does_not_mark_negation(text, tagger):
    assert analyze_jp(text, tagger).negated is False, (
        f"{text!r} must NOT be read as a final-predicate negation "
        f"(false fires corrupt good translations)"
    )


# Grammaticalized / lexicalized ない endings that are semantically AFFIRMATIVE.
# Each was verified as a FALSE FIRE (negated=True) against the real UniDic
# tagger before _is_affirmative_idiom was added; a false fire lets the guard's
# retry invert a correct translation (e.g. "It might rain." -> "It might not
# rain."). Direction-A exclusion family -- see _is_affirmative_idiom.
_GRAMMATICALIZED_AFFIRMATIVE = [
    "雨が降るかもしれない",   # (A) epistemic possibility "it might rain"
    "食べなければいけない",   # (B) obligation "must eat" (なければ + いけない)
    "食べないといけない",     # (B) obligation (ないと breaks/reconnects via と)
    "やらなくてはいけない",   # (B) obligation (なくては; は breaks the cluster)
    "食べなきゃいけない",     # (B) obligation (なきゃ 仮定形-融合)
    "勉強しなければならない",  # (B) obligation (ならない tail)
    "行かねばならない",       # (B) obligation (ねばならない; ぬ/ず gate)
    "行くしかない",           # (C) only-option "have no choice but to go"
    "違いない",               # (D) certainty
    "間違いない",             # (D) certainty "no doubt"
    "しようがない",           # (E) "there's no way / it can't be helped"
    "暑くてたまらない",       # (F) lexicalized 〜てたまらない "unbearably hot"
    # Polite / formal register: a trailing 助動詞 after the いけ/なら+ない or after
    # the ない (ます -> いけません, だ -> だろう, です -> でしょう/です) used to
    # defeat the pred_tokens[-1] negation anchor and let these false-fire. The
    # anchor now looks past the copula/aux tail. Verified false fires vs UniDic.
    "食べなければいけません",   # (B) polite obligation (いけ ませ ん)
    "食べなければなりません",   # (B) polite obligation (なり ませ ん)
    "行かなければいけません",   # (B) polite obligation
    "食べないといけません",     # (B) polite obligation (ないと + いけません)
    "やらなくてはいけません",   # (B) polite obligation (なくては + いけません)
    "勉強しなければなりません",  # (B) polite obligation (なりません tail)
    "間違いないだろう",         # (D) certainty + presumptive だろう
    "間違いないでしょう",       # (D) certainty + polite でしょう
    "間違いないです",           # (D) certainty + polite です
    "行くしかないだろう",       # (C) only-option + だろう
    "行くしかないでしょう",     # (C) only-option + でしょう
    "しようがないだろう",       # (E) can't-be-helped + だろう
    "しようがないでしょう",     # (E) can't-be-helped + でしょう
    "かもしれないでしょう",     # (A) epistemic possibility + でしょう
    # Polite ある/ます-negative register: the idiom's 形容詞 ない is spoken as
    # あり(有る 連用形) + ます + ん (…ありません), or for (F) just ます + ん on the
    # 堪る stem (…たまりません). The final anchor is now the ん(=ず); the effective
    # idiom anchor `ea` re-points past the ます / あり tail so C/D/E/F still see
    # the idiom stem. Verified false fires vs UniDic before the re-point.
    "行くしかありません",       # (C) only-option (しか + ありません)
    "違いありません",           # (D) certainty (違い + ありません)
    "間違いありません",         # (D) certainty "no doubt" (間違い + ありません)
    "しようがありません",       # (E) can't-be-helped (しようが + ありません)
    "暑くてたまりません",       # (F) unbearably hot (堪り + ません, no あり)
    "間違いありませんでした",   # (D) past …ありませんでした (でし/た tail)
    "行くしかありませんね",     # (C) + final particle ね (stripped, still idiom)
    # (G)/(H): the two endings the original A-F enumeration missed. Both
    # false-fired direction A on CORRECT translations ("This could well
    # lead to war." / "I must go." carry no EN negation token), and the
    # hint-misled retry ("might NOT lead to war" / "can't go") PASSES the
    # en_has_negation acceptance gate -- a meaning inversion shipped as a
    # "fix". Seeded defect: delete rule (G) or (H) -> the matching cases
    # below go red.
    "戦争になりかねない",       # (G) 〜かねない "could well happen"
    "事故になりかねません",     # (G) polite (かね + ませ + ん)
    "行かざるを得ない",         # (H) ざるを得ない "have no choice but to"
    "認めざるを得ません",       # (H) polite (得 + ませ + ん)
]


@pytest.mark.parametrize("text", _GRAMMATICALIZED_AFFIRMATIVE)
def test_analyze_jp_does_not_mark_grammaticalized_affirmative(text, tagger):
    assert analyze_jp(text, tagger).negated is False, (
        f"{text!r} is a grammaticalized/idiomatic AFFIRMATIVE ending and must "
        f"NOT be read as a negation (a false fire lets the guard invert a "
        f"correct translation)"
    )


# Interrogative negation: after the 終助詞 strip, 行きませんか was
# token-identical to 行きません (pinned must-fire above), so direction A
# fired on every polite invitation -- one extra llama round-trip per line
# and an accepted retry restyling "Shall we go?" into "Won't you...?".
# A question's polarity is not a statement's; the guard must skip it
# (fail-open: worst case is a skipped retry, never an inversion).
_INTERROGATIVE_NEGATION = [
    "行きませんか",           # polite invitation
    "一緒に飲みませんか",     # polite invitation with lead-in
    "行かないか",             # plain invitation
    # か need not be the literal last token nor sit directly on the marker:
    # trailing 終助詞 may follow it, and a polite です may inflect the ない
    # beneath it. Each is an invitation / wondering, not a negative statement.
    "行きませんかね",         # invitation, ね after か (ん + か + ね, both 終助詞)
    "早く来ないかな",         # wondering/hoping ("I hope they come soon"; か + な)
    "飲みに行かないですか",   # colloquial-polite invitation (ない + です + か)
]


@pytest.mark.parametrize("text", _INTERROGATIVE_NEGATION)
def test_analyze_jp_skips_interrogative_negation(text, tagger):
    """Seeded defect: delete the interrogative gate in analyze_jp -> the
    stripped か leaves a bare ません/ない tail -> negated True -> red."""
    assert analyze_jp(text, tagger).negated is False, (
        f"{text!r} is an invitation/question; firing direction A here "
        f"wastes a retry per line and restyles good translations"
    )


def test_analyze_jp_empty_is_all_false(tagger):
    assert analyze_jp("", tagger) == JpPolarity(False, False, False)


# --------------------------------------------------------------------------
# JP causative / benefactive grant detection (direction B input) + prohibitive
# --------------------------------------------------------------------------
#
# Direction B fires ONLY on the measured class: a causative せて / させて followed
# by a benefactive grant imperative (やって(くれ) / くれ / ください / やれ) -- the
# 死なせてやってくれ -> "Don't let him die" failure. A bare -te request, なさい, or
# plain てください WITHOUT the causative せ/させ is a natural affirmative request
# the model translates fine, so it must NOT be marked (a wasted retry would
# restyle a good line). Fixtures ground-truthed against the real UniDic tagger.

_CAUSATIVE_GRANT = [
    "死なせてやってくれ",   # eval fixture: せ(せる) + て + やって + くれ
    "死なせてやって",       # せて + やって (benefactive くれ dropped)
    "見せてくれ",           # lexical 見せる: surface せて + くれ
    "見せてください",       # せて + ください
    "助けさせてくれ",       # させ(させる) + て + くれ
    "食べさせてやれ",       # させ + て + やれ (命令形 遣る)
]

_NOT_CAUSATIVE_GRANT = [
    "静かにして",           # bare -te request (して, NOT せて) -- over-fired before
    "じっとして",           # bare -te request
    "そのままにして",       # bare -te request
    "待って",               # bare te-form
    "待ってくれ",           # benefactive くれ but NO causative せ/させ
    "手伝ってください",     # ください but NO causative せ/させ
    "早くしなさい",         # なさい, no causative
    "見せてくれる",         # declarative くれる (not the bare imperative くれ)
    "見せてくれた",         # past くれた (not the bare imperative くれ)
    "今日はいい天気です",   # declarative
    "知らない",             # negated declarative
    "行くな",               # prohibitive
]


@pytest.mark.parametrize("text", _CAUSATIVE_GRANT)
def test_analyze_jp_marks_causative_grant(text, tagger):
    assert analyze_jp(text, tagger).causative_grant is True, (
        f"{text!r} should be detected as a causative / benefactive grant request"
    )


@pytest.mark.parametrize("text", _NOT_CAUSATIVE_GRANT)
def test_analyze_jp_does_not_mark_causative_grant(text, tagger):
    assert analyze_jp(text, tagger).causative_grant is False, (
        f"{text!r} must NOT be read as a causative grant "
        f"(over-firing wastes a retry that can restyle a good line)"
    )


@pytest.mark.parametrize("text", ["行くな", "負けるな"])
def test_analyze_jp_marks_prohibitive(text, tagger):
    """Prohibitive final な (verb + な) is negative -- direction B must never
    fire on it (the JP already means 'don't ...', so an EN 'Don't' is correct).
    """
    assert analyze_jp(text, tagger).prohibitive is True, (
        f"{text!r} should be detected as prohibitive"
    )


@pytest.mark.parametrize(
    "text", ["待って", "早くしなさい", "死なせてやってくれ", "今日はいい天気です"]
)
def test_analyze_jp_does_not_mark_prohibitive(text, tagger):
    assert analyze_jp(text, tagger).prohibitive is False, (
        f"{text!r} must NOT be read as prohibitive"
    )


# --------------------------------------------------------------------------
# EN negation-token detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "I do not know.",
        "I don't know.",
        "He won't come.",
        "It isn't ready.",
        "Never again.",
        "No one came.",
        "There is nobody here.",
        "It means nothing.",
        "None of them work.",
        "Neither option is good.",
        "Coffee or tea, but nor milk.",
        "I left without you.",
        "There is no time.",
        "You cannot go.",          # F2: one-word "cannot" (\bnot\b misses it)
        "I am unable to go.",      # F2: unable
        "That's impossible.",      # F2: impossible
    ],
)
def test_en_has_negation_true(text):
    assert en_has_negation(text) is True, f"{text!r} carries a negation token"


@pytest.mark.parametrize(
    "text",
    [
        "I know.",
        "Now go home.",       # "now" must not match \bno\b
        "He said yes.",
        "Let him die.",
        "It works fine.",
        "It is possible.",    # "possible" must not match \bimpossible\b
        "I am able to go.",   # "able" must not match \bunable\b
        "",
    ],
)
def test_en_has_negation_false(text):
    assert en_has_negation(text) is False, (
        f"{text!r} carries no negation token"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Don't let him die.",
        "Do not go.",
        "Stop! Don't touch it.",     # after '! '
        "Please, don't move.",       # after ', '
    ],
)
def test_en_has_added_imperative_negation_true(text):
    assert en_has_added_imperative_negation(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "I don't know.",             # mid-clause declarative, not imperative
        "He won't come.",
        "It's nothing.",
        "Let him die.",
        "",
    ],
)
def test_en_has_added_imperative_negation_false(text):
    assert en_has_added_imperative_negation(text) is False
