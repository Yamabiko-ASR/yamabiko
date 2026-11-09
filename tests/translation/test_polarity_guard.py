"""Translator-level tests for the polarity guard.

These cross the real seam: a fake httpx client returns a scripted sequence of
completions, and the Translator runs its REAL polarity detector (real fugashi
tagger via the injected `tagger` fixture) against them. So a broken detector,
a broken fire condition, or a broken retry-acceptance rule all surface here --
not just a mocked-away "the guard was called".

Seeded-defect drills (one-line production changes that redden a test):
  - force `direction = None` in _apply_polarity_guard
      -> direction-A/B fire tests go red (no retry, original kept).
  - accept the retry unconditionally (drop the flip check)
      -> "retry not flipped keeps original" goes red.
  - loop the semantic retry
      -> "never two extra calls" goes red (calls == 3).
  - ignore cfg.polarity_guard
      -> "disabled makes no extra call" goes red.
"""
from __future__ import annotations

import logging

import httpx
import pytest

import yamabiko.translation.translator as translator_mod
from yamabiko.translation.translator import Translator, TranslationConfig


class _ScriptedClient:
    """Fake httpx.AsyncClient: returns queued responses in order, recording the
    request body and per-request timeout of EVERY post (not just the last) so
    the retry prompt and its time bound can be inspected."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.bodies: list[dict] = []
        self.timeouts: list[object] = []

    async def post(self, url, **kwargs):
        self.calls += 1
        self.bodies.append(kwargs.get("json"))
        self.timeouts.append(kwargs.get("timeout"))
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def aclose(self):
        pass


def _ok(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": text}}]},
        request=httpx.Request("POST", "http://fake"),
    )


def _leak(text: str) -> httpx.Response:
    """A completion whose content carries a <think> block. _complete rejects it
    with a RuntimeError (the thinking-leak guard), so scripting one as the retry
    response drives the guard's retry completion to raise."""
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": f"<think>{text}</think>"}}]},
        request=httpx.Request("POST", "http://fake"),
    )


def _cfg(*, polarity_guard: bool) -> TranslationConfig:
    return TranslationConfig(
        endpoint="http://localhost:8090/v1/chat/completions",
        model="test", max_tokens=40, temperature=0.1,
        polarity_guard=polarity_guard,
    )


def _translator(responses, *, polarity_guard: bool, tagger) -> Translator:
    t = Translator(_cfg(polarity_guard=polarity_guard), tagger=tagger)
    t._client = _ScriptedClient(responses)
    return t


# --------------------------------------------------------------------------
# Direction A: JP negated, EN dropped the negation
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_direction_a_fires_and_accepts_when_negation_restored(tagger):
    t = _translator(
        [_ok("It works."), _ok("It doesn't work.")],
        polarity_guard=True, tagger=tagger,
    )
    result = await t.translate("応用できてない", "anime", [])
    assert result == "It doesn't work.", "the flipped retry must be adopted"
    assert t._client.calls == 2, "exactly one extra (semantic) call"
    assert t.guard_fired == 1
    assert t.retry_accepted == 1


@pytest.mark.asyncio
async def test_direction_a_retry_not_flipped_keeps_original(tagger):
    t = _translator(
        [_ok("It works."), _ok("It totally works.")],  # retry still no negation
        polarity_guard=True, tagger=tagger,
    )
    result = await t.translate("応用できてない", "anime", [])
    assert result == "It works.", "a non-flipping retry must be discarded"
    assert t._client.calls == 2
    assert t.guard_fired == 1
    assert t.retry_accepted == 0


@pytest.mark.asyncio
async def test_direction_a_retry_prompt_names_the_negation_issue(tagger):
    t = _translator(
        [_ok("It works."), _ok("It doesn't work.")],
        polarity_guard=True, tagger=tagger,
    )
    await t.translate("応用できてない", "anime", [])
    retry_user_msg = t._client.bodies[1]["messages"][1]["content"]
    assert "negation" in retry_user_msg.lower(), (
        f"the retry must augment the prompt naming the dropped negation; "
        f"got {retry_user_msg!r}"
    )
    assert "応用できてない" in retry_user_msg, "retry must still carry the raw JP"


@pytest.mark.asyncio
async def test_no_fire_when_en_already_negated(tagger):
    """JP negated AND EN already carries a negation -> polarity agrees ->
    no retry."""
    t = _translator([_ok("It doesn't work.")], polarity_guard=True, tagger=tagger)
    result = await t.translate("応用できてない", "anime", [])
    assert result == "It doesn't work."
    assert t._client.calls == 1
    assert t.guard_fired == 0


# --------------------------------------------------------------------------
# Direction B: JP a plain request, EN invented a "Don't"
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_direction_b_fires_and_removes_added_negation(tagger):
    t = _translator(
        [_ok("Don't let him die, man..."), _ok("Grant him his wish to die.")],
        polarity_guard=True, tagger=tagger,
    )
    result = await t.translate("死なせてやってくれ", "anime", [])
    assert result == "Grant him his wish to die.", (
        "the de-negated retry must be adopted"
    )
    assert t._client.calls == 2
    assert t.guard_fired == 1
    assert t.retry_accepted == 1


@pytest.mark.asyncio
async def test_direction_b_does_not_fire_on_prohibitive(tagger):
    """行くな IS prohibitive ('don't go'), so an EN 'Don't go.' is CORRECT --
    direction B must not fire."""
    t = _translator([_ok("Don't go.")], polarity_guard=True, tagger=tagger)
    result = await t.translate("行くな", "anime", [])
    assert result == "Don't go."
    assert t._client.calls == 1
    assert t.guard_fired == 0


@pytest.mark.asyncio
async def test_direction_b_retry_not_flipped_keeps_original(tagger):
    t = _translator(
        [_ok("Don't let him die."), _ok("Never let him die.")],  # still negated
        polarity_guard=True, tagger=tagger,
    )
    result = await t.translate("死なせてやってくれ", "anime", [])
    assert result == "Don't let him die.", "a still-negated retry is discarded"
    assert t._client.calls == 2
    assert t.retry_accepted == 0


@pytest.mark.asyncio
async def test_direction_b_fires_on_causative_benefactive(tagger):
    """見せてくれ is a causative/benefactive grant (surface せて + くれ); an EN
    'Don't show me.' inverts it, so direction B fires and adopts the de-negated
    retry."""
    t = _translator(
        [_ok("Don't show me."), _ok("Show it to me.")],
        polarity_guard=True, tagger=tagger,
    )
    result = await t.translate("見せてくれ", "anime", [])
    assert result == "Show it to me.", "the de-negated retry must be adopted"
    assert t._client.calls == 2
    assert t.guard_fired == 1
    assert t.retry_accepted == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "jp,en",
    [
        ("静かにして", "Don't make noise."),   # bare -te request; "Don't" is fine
        ("じっとして", "Don't move."),
        ("そのままにして", "Don't touch it."),
    ],
)
async def test_direction_b_does_not_fire_on_bare_te_request(jp, en, tagger):
    """A natural affirmative -te request (して, NOT せて) translated with an
    imperative 'Don't ...' is a CORRECT rendering. Narrowed direction B must not
    fire -- the queued second response must stay unused (no wasted retry that
    could restyle a good line)."""
    t = _translator(
        [_ok(en), _ok("RETRY-MUST-NOT-BE-USED")],
        polarity_guard=True, tagger=tagger,
    )
    result = await t.translate(jp, "anime", [])
    assert result == en, "the original affirmative-request translation is kept"
    assert t._client.calls == 1, "no extra call: bare-te is not the measured class"
    assert t.guard_fired == 0


@pytest.mark.asyncio
async def test_direction_a_does_not_fire_when_en_says_cannot(tagger):
    r"""行けない is negated; EN 'You cannot go.' carries the negation via the
    one-word 'cannot' (which \bnot\b misses). Direction A must NOT fire."""
    t = _translator(
        [_ok("You cannot go."), _ok("RETRY-MUST-NOT-BE-USED")],
        polarity_guard=True, tagger=tagger,
    )
    result = await t.translate("行けない", "anime", [])
    assert result == "You cannot go."
    assert t._client.calls == 1, "no extra call: EN already carries the negation"
    assert t.guard_fired == 0


@pytest.mark.asyncio
async def test_tagger_construction_runs_off_the_event_loop(tagger, monkeypatch):
    """F4: _get_tagger (which may trigger a heavy UniDic load) must run in a
    worker thread, never on the event loop. Reverting the to_thread wrap so it
    runs inline on the loop reddens this (captured thread == loop thread)."""
    import threading

    loop_thread = threading.get_ident()
    captured: dict[str, int] = {}
    real_get = Translator._get_tagger

    def spy(self):
        captured["thread"] = threading.get_ident()
        return real_get(self)

    monkeypatch.setattr(Translator, "_get_tagger", spy)
    t = _translator(
        [_ok("It works."), _ok("It doesn't work.")],
        polarity_guard=True, tagger=tagger,
    )
    await t.translate("応用できてない", "anime", [])
    assert "thread" in captured, "the guard must call _get_tagger"
    assert captured["thread"] != loop_thread, (
        "tagger construction must run off the event loop (in a worker thread)"
    )


# --------------------------------------------------------------------------
# Rhetorical exclusion + disabled + retry budget
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_does_not_fire_on_rhetorical_negation(tagger):
    """いいじゃないか reads as an affirmation despite carrying ない; direction A
    must not fire even though the EN has no negation."""
    t = _translator([_ok("That's great, right?")], polarity_guard=True, tagger=tagger)
    result = await t.translate("いいじゃないか", "variety", [])
    assert result == "That's great, right?"
    assert t._client.calls == 1
    assert t.guard_fired == 0


@pytest.mark.asyncio
async def test_disabled_makes_no_extra_call_and_returns_identical(tagger):
    """With the guard off, a negation-dropping translation is returned as-is:
    zero extra calls, byte-identical output."""
    t = _translator([_ok("It works.")], polarity_guard=False, tagger=tagger)
    result = await t.translate("応用できてない", "anime", [])
    assert result == "It works."
    assert t._client.calls == 1
    assert t.guard_fired == 0


@pytest.mark.asyncio
async def test_never_makes_two_extra_calls(tagger):
    """Retry budget: even when the retry ALSO fails the polarity check, the
    guard makes at most ONE extra completion (never re-retries)."""
    t = _translator(
        [_ok("It works."), _ok("It works great.")],  # both drop the negation
        polarity_guard=True, tagger=tagger,
    )
    result = await t.translate("応用できてない", "anime", [])
    assert result == "It works.", "original kept after the single failed retry"
    assert t._client.calls == 2, "one original + exactly one retry, never more"


@pytest.mark.asyncio
async def test_empty_jp_does_not_invoke_guard(tagger):
    """An empty source line skips the guard entirely (no tagger work, no call
    beyond the base completion)."""
    t = _translator([_ok("")], polarity_guard=True, tagger=tagger)
    result = await t.translate("", "variety", [])
    assert result == ""
    assert t._client.calls == 1
    assert t.guard_fired == 0


# --------------------------------------------------------------------------
# Fail-open: a guard-internal failure must never discard the accepted
# translation. The guard is an optional quality check layered on a completion
# that already succeeded, so an analysis / retry error degrades to "no check",
# not "no subtitle" (main.py scores a raised translate() as a dropped line
# feeding the stage-escalation/restart tracker).
#
# Seeded-defect drill: removing the fail-open try/except in
# _apply_polarity_guard lets the internal error propagate out of translate(),
# reddening tests 1-3 (the awaited translate() raises); incrementing the
# fail-open counter unconditionally reddens the normal-flow test.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyzer_error_keeps_first_translation_and_warns(
    tagger, monkeypatch, caplog
):
    """When the JP polarity analysis raises, the guard fails open: it returns
    the accepted first completion verbatim, makes no retry, and logs one
    greppable WARNING. Reverting the fail-open wrap propagates the error out of
    translate() and reddens this."""

    def _boom(self, jp_text):
        raise ValueError("analysis blew up")

    monkeypatch.setattr(Translator, "_analyze_jp_polarity", _boom)
    t = _translator([_ok("It works.")], polarity_guard=True, tagger=tagger)
    with caplog.at_level(logging.WARNING, logger="yamabiko.translation.translator"):
        result = await t.translate("応用できてない", "anime", [])
    assert result == "It works.", (
        "the accepted translation must survive an analyzer failure"
    )
    assert t._client.calls == 1, "no retry: the failure preceded the retry decision"
    assert t.guard_fired == 0, "the analysis raised before the guard could fire"
    assert t.guard_failed_open == 1
    assert "polarity guard failed open" in caplog.text


@pytest.mark.asyncio
async def test_retry_timeout_keeps_first_translation_after_one_bounded_attempt(
    tagger, monkeypatch, caplog
):
    """The guard's retry is a quality extra on a line that already has a
    translation, so it gets ONE request bounded at 5 s instead of the normal
    two attempts at the client's 30 s timeout plus a 1 s backoff. A retry
    that times out fails open and the accepted first completion is shown.

    Seeded defect: call the retry _complete with the defaults -> a second
    attempt is made (calls == 3) and the retry request carries no 5 s
    bound -> red."""

    async def _no_sleep(*_a, **_k):  # neutralize a backoff if one ever runs
        return None

    monkeypatch.setattr(translator_mod.asyncio, "sleep", _no_sleep)
    t = _translator(
        [
            _ok("It works."),
            httpx.ReadTimeout("retry timed out"),
            _ok("It doesn't work."),  # a second retry attempt would get this
        ],
        polarity_guard=True, tagger=tagger,
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.translation.translator"):
        result = await t.translate("応用できてない", "anime", [])
    assert result == "It works.", (
        "a timed-out retry must not discard the accepted translation"
    )
    assert t._client.calls == 2, "base completion + exactly one retry attempt"
    assert t._client.timeouts[1] == 5.0, "the retry request must be bounded at 5 s"
    assert t.guard_fired == 1, "the guard fired before the retry failed"
    assert t.guard_failed_open == 1
    assert "polarity guard failed open" in caplog.text


@pytest.mark.asyncio
async def test_retry_runtime_error_keeps_first_translation(tagger, caplog):
    """A thinking-leak on the retry completion raises RuntimeError out of
    _complete (not a transient error, so no transport retry). That guard-internal
    failure must fail open and keep the accepted first completion."""
    t = _translator(
        [_ok("It works."), _leak("secret reasoning")],
        polarity_guard=True, tagger=tagger,
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.translation.translator"):
        result = await t.translate("応用できてない", "anime", [])
    assert result == "It works.", (
        "a thinking-leak retry failure must not discard the accepted translation"
    )
    assert t._client.calls == 2, "base completion + one retry that raised on the leak"
    assert t.guard_fired == 1
    assert t.guard_failed_open == 1
    assert "polarity guard failed open" in caplog.text


@pytest.mark.asyncio
async def test_normal_guarded_flow_does_not_increment_fail_open(tagger):
    """A healthy guarded flow (fire, retry succeeds, flip accepted) is not a
    fail-open: the counter stays 0. Incrementing guard_failed_open on the happy
    path reddens this."""
    t = _translator(
        [_ok("It works."), _ok("It doesn't work.")],
        polarity_guard=True, tagger=tagger,
    )
    result = await t.translate("応用できてない", "anime", [])
    assert result == "It doesn't work."
    assert t.guard_fired == 1
    assert t.retry_accepted == 1
    assert t.guard_failed_open == 0, "a successful guarded flow is not a fail-open"
