"""Polarity (negation) analysis for the translator's polarity guard.

Pure functions with no I/O. The Japanese-side analysis needs a fugashi /
UniDic tagger for POS + inflection features; the caller injects it (same
pattern as ``ProperNounExtractor``) so this module never loads a dictionary
itself. The English-side checks are regex-only.

Detection is deliberately CONSERVATIVE. A missed fire degrades nothing (the
translator keeps its original output), but a false fire triggers a retry that
can corrupt a good translation, so every rule errs toward not-firing:
  - only a negation in the FINAL predicate chain counts (a relative-clause
    negation like 食べない人が来た is affirmative overall);
  - obligation idioms that are grammatically negative but semantically
    affirmative (行かねば / 行かなきゃ / 行かないと = "gotta go") are excluded via
    the 仮定形 / clause-break rules;
  - grammaticalized / lexicalized ない endings that read AFFIRMATIVE
    (かもしれない = "might", 食べなければいけない = "must eat", 行くしかない =
    "have no choice but to go", 間違いない = "no doubt", しようがない =
    "can't be helped", 暑くてたまらない = "unbearably hot") are excluded as a
    unit -- see ``_is_affirmative_idiom``;
  - rhetorical / copular ...じゃない endings (いいじゃないか = "that's good,
    right") never count as a negation to preserve.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# --------------------------------------------------------------------------
# English side (regex only)
# --------------------------------------------------------------------------

# Word-boundary negative tokens. "no" as a bare determiner (followed by
# whitespace) is included per spec. Over-detecting EN negation is the SAFE
# error: for direction A it means the guard does not fire (a miss), and for
# the acceptance flip a spurious negative only makes us keep the original.
_EN_NEG = re.compile(
    r"""(?xi)
    \b (?: not | never | nobody | nothing | none | neither | nor | without
         | cannot | unable | impossible ) \b
    | \b no \s+ one \b
    | \b no - one \b
    | n ['’] t \b          # contracted n't: don't, can't, isn't, won't ...
    | \b no \b (?= \s )         # determiner "no " (no time, no way)
    """
)


def en_has_negation(text: str) -> bool:
    """True if the English carries any surface negation token."""
    return bool(text) and _EN_NEG.search(text) is not None


# Imperative "Don't" / "Do not": sentence-initial or right after clause
# punctuation. Deliberately does NOT match a mid-clause "... I don't know ..."
# (a declarative), which would false-fire direction B. "Please don't" without a
# comma is a tolerated miss (conservative).
_EN_IMPERATIVE_DONT = re.compile(
    r"(?:^|[.!?,;]\s+)(?:do\s+not|don['’]t)\b",
    re.IGNORECASE,
)


def en_has_added_imperative_negation(text: str) -> bool:
    """True if the English opens a clause with an imperative Don't / Do not."""
    return bool(text) and _EN_IMPERATIVE_DONT.search(text) is not None


# --------------------------------------------------------------------------
# Japanese side (needs a fugashi / UniDic tagger)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class JpPolarity:
    negated: bool          # clear negation in the FINAL predicate chain
    causative_grant: bool  # causative/benefactive grant request (see below)
    prohibitive: bool      # prohibitive final な (行くな = "don't go")


# Negation auxiliary / adjective lemmas. ん(=ぬ) surfaces with lemma ず in
# UniDic (わかりません -> ん lemma=ず; わからん -> ん lemma=ず); literary ず/ぬ
# (やめておらず -> ず lemma=ず). The 形容詞 branch covers 無い/ない existential
# negation (元気がない) and the copular ...じゃない (excluded as rhetorical below).
_NEG_AUX_LEMMAS = {"ない", "ぬ", "ず"}
_NEG_ADJ_LEMMAS = {"ない", "無い"}

# Only a negation in a sentence-FINAL inflection counts. 仮定形 (行かねば /
# 行かなければ / 行かなきゃ) is the "if-not -> must" obligation idiom (a false-fire
# risk), and 連体形 marks a relative-clause modifier; requiring 終止形 or 連用形
# excludes both.
_FINAL_CFORMS = ("終止形", "連用形")

_PREDICATE_POS1 = {"動詞", "形容詞", "助動詞"}

# Rhetorical / copular-negation tag endings. じゃないか / ではないか / じゃん etc.
# read as affirmations ("it's good, isn't it") despite carrying ない, so they
# must NOT count as a negation to preserve. A bare copular ...じゃない /
# ...ではない is swallowed here too (conservative: a real "X is not Y" is skipped
# rather than risk a rhetorical いいじゃない false fire, which has no trailing か
# in the text yet reads as affirmation).
_RHETORICAL_TAG = re.compile(
    r"(?:じゃ|では|じゃあ)ない(?:です)?(?:か|の|ん)?(?:ね|よ|かな|の)?"
    r"[。、．！？!?.…\s]*[」』）)】]*\s*$"
    r"|じゃん[。、．！？!?.…\s]*[」』）)】]*\s*$"
)

# Causative / benefactive grant request -- direction B's ONLY trigger. A
# causative せて / させて somewhere in the FINAL predicate chain, followed by a
# benefactive grant imperative tail: やって(くれ) / くれ / ください / やれ. This is
# the MEASURED class that fools the model into an inverted "Don't ..."
# (死なせてやってくれ -> "Don't let him die"; 見せてくれ -> "Don't show me").
#
# Why this exact shape and nothing wider:
#   - the せて/させて requirement excludes bare -te requests (静かにして -> して,
#     no せて), plain てください / てくれ without a causative (手伝ってください,
#     待ってくれ), and なさい -- all natural affirmative requests the model renders
#     correctly, where an extra retry would only restyle a good line;
#   - the trailing $ anchor keeps the tail to the BARE imperative grants:
#     declarative 見せてくれる and past 見せてくれた run past the くれ, so くれ$ /
#     ください$ / やれ$ / やって$ do not match them.
# Matched against the reconstructed chain surface (a contiguous sentence suffix),
# so a non-final せて elsewhere in the line cannot trigger it.
_CAUSATIVE_GRANT = re.compile(
    r"(?:せて|させて).*?(?:やってくれ|やって|くれ|ください|やれ)$"
)


def _pos1(tok) -> str:
    return getattr(tok.feature, "pos1", "") or ""


def _pos2(tok) -> str:
    return getattr(tok.feature, "pos2", "") or ""


def _cform(tok) -> str:
    return getattr(tok.feature, "cForm", "") or ""


def _lemma(tok) -> str:
    f = tok.feature
    return getattr(f, "lemma", None) or getattr(f, "orthBase", None) or tok.surface


def _is_final_neg_marker(tok) -> bool:
    if not _cform(tok).startswith(_FINAL_CFORMS):
        return False
    lem = _lemma(tok)
    p1 = _pos1(tok)
    if p1 == "助動詞" and lem in _NEG_AUX_LEMMAS:
        return True
    if p1 == "形容詞" and lem in _NEG_ADJ_LEMMAS:
        return True
    return False


def _strip_trailing_punct(tokens):
    i = len(tokens)
    while i > 0 and _pos1(tokens[i - 1]) == "補助記号":
        i -= 1
    return tokens[:i]


def _strip_trailing_final_particles(tokens):
    """Drop trailing sentence-final particles (よ ね な か の ...). Tokens are
    assumed already punctuation-stripped."""
    i = len(tokens)
    while (
        i > 0
        and _pos1(tokens[i - 1]) == "助詞"
        and _pos2(tokens[i - 1]) == "終助詞"
    ):
        i -= 1
    return tokens[:i]


def _final_predicate_cluster(tokens):
    """The trailing run of predicate tokens (verb / adjective / auxiliary plus
    connective 接続助詞 particles), walking back from the end and stopping at
    the first non-predicate token (noun, case / binding particle, adverb ...).

    This isolates the FINAL predicate so a subordinate-clause negation
    (食べない人が来た, where ない modifies 人 across the case particle が) does not
    count, and an obligation idiom (行かないと, where と is a 格助詞 that breaks
    the cluster) is excluded too.
    """
    out = []
    for tok in reversed(tokens):
        p1 = _pos1(tok)
        if p1 in _PREDICATE_POS1 or (p1 == "助詞" and _pos2(tok) == "接続助詞"):
            out.append(tok)
        else:
            break
    return out


# Lemmas that read as a negation in the gate scans below.
_NEG_LEMMAS = _NEG_AUX_LEMMAS | _NEG_ADJ_LEMMAS


def _surface(tok) -> str:
    return getattr(tok, "surface", "") or ""


def _final_neg_anchor(pred_tokens) -> int:
    """Index of the would-be final negation marker (a 終止形/連用形 ない・無い・
    ん・ず) in the predicate tail, looking PAST any trailing non-negation
    copula/aux 助動詞 that inflect it further without changing its polarity --
    polite ます (…ません), the copula だ / です, and the presumptive
    だろう / でしょう / う (…ないだろう / …ないでしょう / …ないです).

    Returns -1 when the tail does not end in such a (possibly aux-suffixed)
    negation marker. Anchoring on this index rather than on ``pred_tokens[-1]``
    is what keeps the idiom rules below firing for the polite / formal register
    (…いけません, …ないでしょう) as well as the plain forms.
    """
    i = len(pred_tokens) - 1
    while i >= 0:
        if _is_final_neg_marker(pred_tokens[i]):
            return i
        # A trailing だろう / でしょう / です / だ inflects the ない beneath it;
        # skip it and keep looking for the real negation anchor.
        if _pos1(pred_tokens[i]) == "助動詞":
            i -= 1
            continue
        return -1
    return -1


def _is_affirmative_idiom(pred_tokens) -> bool:
    """True when the final predicate is a grammaticalized / lexicalized ない
    ending that is semantically AFFIRMATIVE, so its trailing ない must NOT be
    read as a polarity negation.

    Every rule is anchored on the would-be final negation marker (a 終止形/連用形
    ない・無い・ん, located by ``_final_neg_anchor`` past any trailing copula/aux
    助動詞) and inspects only the contiguous predicate tail, so a non-final
    occurrence elsewhere in the line cannot trigger it. Each rule is a precise
    token pattern (lemma / POS), not a surface substring, and is deliberately
    narrow so plain ~ない / ~なかった / ~ません negations and potential negatives
    (行けない / 言えない) keep firing:

      (A) かもしれない  -- epistemic possibility (...か も しれ ない[でしょう]).
      (B) obligation chain -- <negative-conditional> ... いけ|なら [ます] + ない/ん
          ("must / have to": 食べなければいけない, やらなくてはいけない,
          食べないといけない, 勉強しなければならない, and the polite
          …いけません / …なりません). Requires an EARLIER negation feeding a
          conditional connective into the trailing いけ/なら -- a DOUBLE negative
          that reads affirmative. A SINGLE negative (食べてはいけない /
          食べてはいけません = "must not eat") and a standalone それはいけない
          (no earlier negation) both keep firing. The いけ/なら verb can be
          separated from the final ん by a polite ます, so walk the inflection
          tail back over any 助動詞 to reach it.
      (C) V/N しか + ない  -- only-option idiom (行くしかない = "have to go").
      (D) 違い / 間違い + ない  -- certainty (間違いない = "no doubt").
      (E) 仕様 / 仕方 + が + ない  -- "can't be helped" (元気がない, a real
          existential negation, keeps firing: the noun lemma is whitelisted).
      (F) 堪る + ない  -- lexicalized 〜てたまらない ("unbearably ...").

    Rules (C)-(F) also cover the POLITE ある/ます-negative register, where the
    idiom's ない is spoken as a token chain instead of a single 形容詞: the
    existential ある's 連用形 あり(有る 非自立可能) + polite ます + the ん(=ず) that
    ``_final_neg_anchor`` lands on (行くしかありません, 間違いありません,
    しようがありません, and past …ありませんでした), or -- for (F), whose stem is a
    real verb -- just ます + ん on the 堪る 連用形 (暑くてたまりません). ``ea`` below
    re-points the idiom anchor past that ます (and past the existential あり) so
    each rule sees the same ``prev`` it does in the plain ない form. The re-point
    is anchor-gated: a standalone polite negation (行きません, 知りません) and a
    plain existential …が ありません (お金がありません, 時間がありません) are
    re-pointed too but match no idiom anchor, so they keep firing.
    """
    a = _final_neg_anchor(pred_tokens)
    if a < 1:
        return False

    # Effective idiom anchor. Defaults to the ない anchor `a`; in the polite
    # ある/ます-negative register it walks back over the polite ます (…ませ ん) and,
    # when the predicate verb beneath it is the existential ある (…あり ませ ん),
    # one further, so ``prev`` is the idiom stem exactly as in the plain form.
    # (B) keeps using `a`: it has its own ます-walk for the …いけ/なり ません tail.
    ea = a
    if _pos1(pred_tokens[a - 1]) == "助動詞" and _lemma(pred_tokens[a - 1]) == "ます":
        ea = a - 1
        if (
            ea >= 1
            and _lemma(pred_tokens[ea - 1]) == "有る"
            and _pos2(pred_tokens[ea - 1]) == "非自立可能"
        ):
            ea -= 1
    if ea < 1:
        return False

    prev = pred_tokens[ea - 1]
    prev_lemma = _lemma(prev)
    prev_surface = _surface(prev)

    # (A) かもしれない: しれ(知れる) + ない, gated on a preceding 係助詞 も so a
    # bare 知れない (rare, e.g. 底が知れない) is left alone.
    if prev_lemma == "知れる" and ea >= 2 and _surface(pred_tokens[ea - 2]) == "も":
        return True

    # (F) 〜てたまらない / 〜てたまりません: 堪る + ない (or + ます + ん).
    if prev_lemma == "堪る":
        return True

    # (G) 〜かねない / 〜かねません: 兼ねる + ない -- "could well happen"
    # (戦争になりかねない = "this could well lead to war"). The auxiliary
    # かねる is tagged 動詞/非自立可能 or 接尾辞/動詞的 depending on the
    # UniDic build (unidic-lite, the bundled dictionary, uses 接尾辞);
    # either gate keeps a standalone main-verb 兼ねる + ない (the literal
    # "does not double as", tagged 動詞/一般) out of the exclusion.
    if prev_lemma == "兼ねる" and (
        _pos2(prev) == "非自立可能" or _pos1(prev) == "接尾辞"
    ):
        return True

    # (H) 〜ざるを得ない / 〜ざるを得ません: ざる(=ず) + を + 得る + ない --
    # "have no choice but to" (行かざるを得ない = "I must go"). Gated on the
    # ざる+を prefix so a genuine potential negation on 得る alone
    # (あり得ない "impossible") keeps firing.
    if (
        prev_lemma == "得る"
        and ea >= 3
        and _surface(pred_tokens[ea - 2]) == "を"
        and _lemma(pred_tokens[ea - 3]) == "ず"
    ):
        return True

    # (C) 〜しかない / 〜しかありません: 副助詞 しか directly before the ない / あり.
    if prev_surface == "しか" and _pos1(prev) == "助詞":
        return True

    # (D) 違いない / 間違いない (…ありません).
    if prev_lemma in ("違い", "間違い"):
        return True

    # (E) しようがない / しかたがない (…がありません): 格助詞 が + a whitelisted
    # "way/method" noun. 元気がない, 時間がない, お金がありません, ... are NOT
    # whitelisted, so they still fire.
    if (
        prev_surface == "が"
        and _pos1(prev) == "助詞"
        and ea >= 2
        and _lemma(pred_tokens[ea - 2]) in ("仕様", "仕方")
    ):
        return True

    # (B) obligation chain: trailing いけ(行く) / なら(成る) 非自立可能, preceded
    # -- across any て/と/ば 接続助詞 and the 係助詞 は -- by a negation. The
    # earlier negation is what flips the sense affirmative; without it (それは
    # いけない, 食べてはいけない) the ない keeps firing. In the polite register the
    # verb is followed by ます before the ん (いけ ませ ん), so walk the inflection
    # tail back over any 助動詞 to reach the いけ/なら verb first.
    v = a - 1
    while v >= 0 and _pos1(pred_tokens[v]) == "助動詞":
        v -= 1
    if (
        v >= 0
        and _lemma(pred_tokens[v]) in ("行く", "成る")
        and _pos2(pred_tokens[v]) == "非自立可能"
    ):
        i = v - 1
        while i >= 0 and _pos1(pred_tokens[i]) == "助詞" and (
            _pos2(pred_tokens[i]) == "接続助詞"
            or _surface(pred_tokens[i]) == "は"
        ):
            i -= 1
        if i >= 0:
            gate = pred_tokens[i]
            if _pos1(gate) in ("助動詞", "形容詞") and _lemma(gate) in _NEG_LEMMAS:
                return True

    return False


def analyze_jp(text: str, tagger) -> JpPolarity:
    """Analyse a Japanese line's final-predicate polarity.

    ``tagger`` is a fugashi/UniDic ``Tagger`` (injected; this module loads no
    dictionary). Returns a ``JpPolarity`` triple used by the translator's guard.
    """
    if not text:
        return JpPolarity(False, False, False)
    tokens = list(tagger(text))
    no_punct = _strip_trailing_punct(tokens)

    # Prohibitive final な: a verb directly followed by the 終助詞 な (行くな,
    # 負けるな). Checked on the punctuation-stripped tail BEFORE final particles
    # are removed, since な IS the final particle here.
    prohibitive = False
    if len(no_punct) >= 2:
        last, prev = no_punct[-1], no_punct[-2]
        if (
            last.surface == "な"
            and _pos1(last) == "助詞"
            and _pos2(last) == "終助詞"
            and _pos1(prev) == "動詞"
        ):
            prohibitive = True

    pred_tokens = _strip_trailing_final_particles(no_punct)

    # Interrogative negation (…ませんか / …ないか): after the 終助詞 strip
    # above, 行きませんか is token-identical to 行きません, so direction A
    # fired on every polite invitation -- one wasted retry per line, and
    # an accepted retry restyles "Shall we go?" into "Won't you…?". A
    # question's polarity is not a statement's; skip the flag when an
    # interrogative か sits anywhere in the trailing 終助詞 run and the
    # predicate beneath that run ends in the negation marker. Further 終助詞
    # may follow the か (行きませんかね, 早く来ないかな) and a polite です may
    # inflect the ない beneath it (飲みに行かないですか: ない です か), so the
    # walk skips non-negation 助動詞 on its way to the marker.
    #
    # The walk stops at anything else -- in particular at the 準体助詞 の/ん of
    # 行かないのか / 行かないんですか, which nominalizes a 連体形 negation and
    # marks a GENUINE negative question ("is it that you're not going?"): its
    # correct EN carries the negation, so that shape must set negated instead.
    # The cluster scan alone cannot see it (the 助詞 の breaks the cluster and
    # the 連体形 ない fails _is_final_neg_marker), hence the dedicated flag.
    interrogative_neg = False
    neg_question = False
    if any(_surface(t) == "か" for t in no_punct[len(pred_tokens):]):
        j = len(pred_tokens) - 1
        while (
            j >= 0
            and _pos1(pred_tokens[j]) == "助動詞"
            and not _is_final_neg_marker(pred_tokens[j])
        ):
            j -= 1
        if j >= 0:
            if _is_final_neg_marker(pred_tokens[j]):
                interrogative_neg = True
            elif (
                _pos2(pred_tokens[j]) == "準体助詞"
                and j >= 1
                and _pos1(pred_tokens[j - 1]) in ("助動詞", "形容詞")
                and _lemma(pred_tokens[j - 1]) in _NEG_LEMMAS
            ):
                neg_question = True
    cluster = _final_predicate_cluster(pred_tokens)

    # Negation: a final-form negation marker inside the final predicate cluster
    # (or the nominalized negative question detected above), unless the ending
    # is a rhetorical / copular ...じゃない tag (いいんじゃないですか stays
    # excluded even with its trailing か) or a grammaticalized ない idiom that
    # reads affirmative (かもしれない, 〜なければいけない, 行くしかない,
    # 間違いない, しようがない, 〜てたまらない, ...).
    negated = False
    if (
        not interrogative_neg
        and not _RHETORICAL_TAG.search(text)
        and not _is_affirmative_idiom(pred_tokens)
    ):
        negated = neg_question or any(_is_final_neg_marker(t) for t in cluster)

    # Causative / benefactive grant request: a causative せて / させて followed by
    # a benefactive grant imperative in the FINAL predicate chain. Reconstruct
    # the chain surface (a contiguous sentence suffix) and match the pattern.
    # This is deliberately NARROW -- see _CAUSATIVE_GRANT: bare-te requests,
    # なさい, and plain てください do NOT count.
    causative_grant = False
    if cluster:
        chain_surface = "".join(tok.surface for tok in reversed(cluster))
        causative_grant = _CAUSATIVE_GRANT.search(chain_surface) is not None

    return JpPolarity(
        negated=negated,
        causative_grant=causative_grant,
        prohibitive=prohibitive,
    )
