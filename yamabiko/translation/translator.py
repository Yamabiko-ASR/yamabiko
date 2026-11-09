"""Async translator client — talks to llama.cpp's OpenAI-compatible endpoint."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from yamabiko import loopback_http

from .negation import (
    JpPolarity,
    analyze_jp,
    en_has_added_imperative_negation,
    en_has_negation,
)
from .postprocess import clean
from .prompt import SYSTEM_PROMPT, build_user_message

_log = logging.getLogger(__name__)

# Retry instructions appended to the user message when the polarity guard fires.
# Wording mirrors the two directions the guard detects.
_POLARITY_RETRY_HINT_A = (
    "IMPORTANT: the Japanese line contains a negation; your previous attempt "
    "dropped it. Preserve the negation."
)
_POLARITY_RETRY_HINT_B = (
    "IMPORTANT: the Japanese line does NOT contain a negation; do not add one."
)

# Catch the PARENT classes, not an enumeration of leaves. An
# enumeration of leaf types (ConnectError, ReadTimeout, WriteTimeout,
# NetworkError) misses sibling subtypes that fire under real conditions:
# ConnectTimeout / PoolTimeout (TimeoutException subtypes) and
# RemoteProtocolError (server closed the connection mid-response, e.g.
# during a llama-server restart) would then escape the retry loop on
# attempt 0. Deliberately NOT the broader httpx.TransportError: that would
# also retry LocalProtocolError (a malformed request WE built — a client
# bug retrying can't fix), UnsupportedProtocol, and ProxyError.
_TRANSIENT_ERRORS = (
    httpx.TimeoutException,     # ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout
    httpx.NetworkError,         # ConnectError, ReadError, WriteError, CloseError
    httpx.RemoteProtocolError,  # server violated protocol / dropped mid-response
)

# Per-poll cap for the cheap /health readiness probe — far below the client's
# 30 s default so the startup_readiness_timeout_s budget is honored even if a
# mid-load llama accepts a connection but stalls. A spurious timeout here is
# harmless: it surfaces as httpx.RequestError -> "still loading" -> re-poll.
_READINESS_PROBE_TIMEOUT_S = 5.0


class LlamaNotReadyError(RuntimeError):
    """llama-server did not reach a serving state within the startup budget."""


@dataclass
class TranslationConfig:
    endpoint: str
    model: str
    max_tokens: int
    temperature: float
    startup_readiness_timeout_s: float = 90.0
    startup_readiness_poll_s: float = 2.0
    # When true, translate() runs a negation-preservation check after the normal
    # translation and, if JP/EN polarity disagrees, makes ONE extra llama call to
    # fix it (see yamabiko/translation/negation.py). Off by default so the guard
    # is an opt-in feature; the shipped default_config.yaml enables it.
    polarity_guard: bool = False


class Translator:
    """Async client for llama.cpp's OpenAI-compatible chat endpoint.

    Optional `pipeline` (RomanizationPipeline) is invoked pre-translation:
      - pre: extracts proper-noun candidates and looks them up in the active
        series gazetteer / user glossary, returning a soft "Use these exact
        English names..." hint for the confirmed names. The hint is injected
        via build_user_message; the source is never mutated (translate sends
        its own jp_text unchanged). There is no post pass -- soft hints are
        prompt-only, so the LLM output is returned as-is.
    """

    def __init__(
        self,
        cfg: TranslationConfig,
        pipeline: Any | None = None,
        tagger: Any | None = None,
    ):
        self._cfg = cfg
        self._pipeline = pipeline
        self._client = loopback_http.async_client(cfg.endpoint, timeout=30.0)
        # Tagger for the polarity guard's JP analysis. Precedence:
        #   1. an explicitly injected tagger (tests),
        #   2. the romanization pipeline's tagger (reuse the UniDic already
        #      loaded for romanization -- no second dictionary load),
        #   3. a lazily-built own tagger, only if the guard actually runs while
        #      no pipeline is present (romanization disabled + guard enabled).
        self._injected_tagger = tagger
        self._own_tagger: Any | None = None
        # Counters exposed for the eval harness / DEBUG visibility.
        self.guard_fired = 0
        self.retry_accepted = 0
        # Fail-open events: a guard-internal failure (JP analysis or the ONE
        # retry completion raising) that was swallowed so the already-accepted
        # translation survived. Counted separately from guard_fired so a rise in
        # swallowed failures stays observable. Exposed like its siblings.
        self.guard_failed_open = 0

    @property
    def health_check_url(self) -> str:
        """Return the llama-server base + `/v1/models` for liveness probing.

        The configured endpoint is the chat-completions path (e.g.
        `http://localhost:8090/v1/chat/completions`); stripping the
        path and appending `/v1/models` yields a cheap, OpenAI-spec
        read endpoint suitable for a periodic reachability check from
        HealthProbes in the pipeline's /healthz handler.
        """
        p = urlparse(self._cfg.endpoint)
        return f"{p.scheme}://{p.netloc}/v1/models"

    @property
    def readiness_url(self) -> str:
        """llama-server base + `/health` — the cheap readiness endpoint.

        llama.cpp returns 503 (`{"status":"loading model"}`) while loading the
        GGUF, then 200 once serving. Derived from the chat-completions endpoint
        the same way as `health_check_url`.
        """
        p = urlparse(self._cfg.endpoint)
        return f"{p.scheme}://{p.netloc}/health"

    async def wait_until_ready(self, *, timeout_s: float, poll_s: float) -> None:
        """Block until llama-server's /health serves 200, or raise LlamaNotReadyError.

        Startup-only readiness gate. The watchdog spawns llama and the
        pipeline concurrently and only waits for the model FILE in the cache, so
        llama may still be loading the multi-GB GGUF when build_pipeline runs.
        Status policy:
          - 200 -> ready, return.
          - 404/405 -> this build has no /health -> return; sanity_check gates.
          - ANY 5xx (500-599) and connection-level errors / timeouts ->
            still loading -> keep polling until timeout_s, then raise.
            llama itself serves 503 ("loading model") while loading the
            GGUF, but gateways in front of a loading backend return
            502/504 (nginx/traefik with the upstream not accepting yet);
            bailing on the first 502 skipped the gate and dumped a normal
            5-20 s cold start onto sanity_check's 2-attempt/1-s retry ->
            exit 43 on a setup that cold-started fine.
          - Anything else (3xx redirect, 401/403 auth, ...) -> not a
            loading signal. External endpoints are supported, and one may
            lock down or redirect /health while serving chat completions
            fine; polling such a status to the deadline turned a working
            endpoint into exit 43. Log a warning naming the status and
            return -- sanity_check remains the arbiter (the pre-F113
            behavior for such endpoints).
        Cheap: no token generation.
        """
        url = self.readiness_url
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                r = await self._client.get(url, timeout=_READINESS_PROBE_TIMEOUT_S)
                if r.status_code == 200:
                    return
                if r.status_code in (404, 405):
                    _log.info(
                        "llama /health absent (HTTP %d); skipping readiness gate", r.status_code,
                    )
                    return
                if not 500 <= r.status_code < 600:
                    # 3xx/4xx (auth proxy 401/403, redirect): not a loading
                    # signal. An external endpoint may gate /health yet serve
                    # chat completions. Skip the gate; sanity_check arbitrates.
                    _log.warning(
                        "llama /health returned unexpected HTTP %d; skipping readiness "
                        "gate (sanity_check will verify the endpoint)", r.status_code,
                    )
                    return
                # 5xx = still-loading class: llama itself serves 503 ("loading
                # model") while the GGUF loads, and a gateway in front of a
                # loading backend (nginx/traefik) returns 502/504. Keep polling
                # to the deadline.
            except httpx.RequestError as exc:
                # ConnectError / ConnectTimeout / ReadTimeout etc. — not up yet.
                _log.debug("llama readiness poll error (%s); retrying", type(exc).__name__)
            if time.monotonic() >= deadline:
                raise LlamaNotReadyError(
                    f"llama-server did not become ready at {url} within {timeout_s:.0f}s"
                )
            await asyncio.sleep(poll_s)

    def _build_body(self, user_msg: str) -> dict[str, Any]:
        return {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": self._cfg.max_tokens,
            "temperature": self._cfg.temperature,
            "stream": False,  # non-streaming; SSE response parsing not implemented
            # Disable Qwen3-family thinking the clean way: the chat
            # template switches to non-thinking mode rather than truncating it.
            # Qwen3.5 (both quant tiers) otherwise leaks reasoning /
            # "(Note: ...)" meta
            # into the output and burns latency on thinking tokens. llama.cpp
            # applies this via --jinja (on by default). Harmless to a non-Qwen3
            # model (its template ignores the kwarg). The earlier
            # --reasoning-budget 0 truncation hurt translation quality; this
            # does not. The postprocess clean() backstop catches any residual.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    async def _complete(
        self,
        body: dict[str, Any],
        *,
        timeout_s: float | None = None,
        attempts: int = 2,
    ) -> str:
        """POST one chat-completion body and return the cleaned translation.

        Owns the transport-retry loop (`attempts` tries, 1 s backoff on
        transient network + 5xx errors) and the thinking-leak guard. Kept
        separate from translate() so the polarity guard can request a SECOND
        semantic completion that composes with -- rather than multiplies --
        this transport retry. A non-None `timeout_s` bounds each request
        instead of the client's 30 s default; the guard's retry passes 5 s and
        one attempt, so a stalled server cannot hold an already-translated
        line for another minute.
        """
        # TODO: expose retries + backoff_s on TranslationConfig. 2 attempts /
        # 1 s sleep doesn't cover llama-server cold restarts (5-20 s reload time).
        backoff_s = 1.0
        request_kwargs: dict[str, Any] = {"json": body}
        if timeout_s is not None:
            request_kwargs["timeout"] = timeout_s
        last_exc: Exception | None = None
        for attempt in range(attempts):
            can_retry = attempt < attempts - 1
            try:
                r = await self._client.post(self._cfg.endpoint, **request_kwargs)
                if 500 <= r.status_code < 600 and can_retry:
                    _log.info(
                        "translator got HTTP %d from %s; retrying once",
                        r.status_code, self._cfg.endpoint,
                    )
                    await asyncio.sleep(backoff_s)
                    continue
                r.raise_for_status()
                data = r.json()
                # llama.cpp's OpenAI server can return content: null on a
                # truncated/empty completion (some finish_reason: length).
                # Coerce to "" so the '<think>' check below doesn't raise
                # TypeError; an empty completion then flows through clean() ->
                # "" -> the normal drop path (and sanity_check won't false-fail).
                message = data["choices"][0]["message"]
                raw = message.get("content") or ""
                # Two ways Qwen3 thinking can leak past enable_thinking=False:
                # as a literal <think> block in message.content (a server with
                # no reasoning parsing), or split out into
                # message.reasoning_content (--reasoning-format auto, the
                # upstream + our spawn default, or an external endpoint we don't
                # control). Guard BOTH. Under auto a genuine non-thinking
                # response has reasoning_content=None (verified), so this only
                # fires on a real leak -- it does NOT false-trip on Qwen3's
                # empty non-thinking think-block, which auto parses away.
                if "<think>" in raw or (message.get("reasoning_content") or ""):
                    raise RuntimeError(
                        "Thinking-mode block detected in translation output. Check llama-server flags and system prompt."
                    )
                # Soft hints are prompt-only, so the LLM output needs no
                # post pass -- clean() is the only transform.
                return clean(raw)
            except _TRANSIENT_ERRORS as exc:
                last_exc = exc
                if can_retry:
                    _log.info(
                        "translator network error (%s) calling %s; retrying once",
                        type(exc).__name__, self._cfg.endpoint,
                    )
                    await asyncio.sleep(backoff_s)
                    continue
                raise

        # Should be unreachable, but satisfies type checker
        assert last_exc is not None
        raise last_exc

    def _get_tagger(self) -> Any:
        """Fugashi/UniDic tagger for the polarity guard -- see __init__ for the
        precedence. Lazily builds an own tagger only as the last resort so a
        disabled guard never loads UniDic.

        Single-consumer contract: the guard runs at most one translate() at a
        time per Translator, so the lazy `_own_tagger` init below has no
        concurrent writer even though `_analyze_jp_polarity` calls this from a
        worker thread."""
        if self._injected_tagger is not None:
            return self._injected_tagger
        if self._pipeline is not None:
            return self._pipeline.tagger
        if self._own_tagger is None:
            import fugashi
            self._own_tagger = fugashi.Tagger()
        return self._own_tagger

    def _analyze_jp_polarity(self, jp_text: str) -> JpPolarity:
        """Get-or-build the tagger and run the JP polarity analysis.

        Both the potentially-heavy lazy tagger construction (`_get_tagger`'s
        last-resort UniDic load) and the CPU-bound MeCab parse happen here, so
        the whole thing runs in ONE worker thread and never stalls the event
        loop. See `_get_tagger` for the single-consumer lazy-init contract."""
        return analyze_jp(jp_text, self._get_tagger())

    async def _apply_polarity_guard(
        self, jp_text: str, user_msg: str, result: str
    ) -> str:
        """Run the polarity check over an already-accepted translation, failing
        OPEN on any internal error.

        Invariant: a guard failure must never discard the accepted translation.
        The guard is an optional quality check layered on a completion that
        already succeeded, so an exception from the JP analysis or from the ONE
        retry completion degrades to "no check", not "no subtitle". This matters
        because the translation stage scores a raised translate() as a dropped
        line that also feeds the stage-escalation/restart tracker. First-
        completion failures are raised in translate() before this method runs,
        so a real translation failure still propagates untouched.
        """
        try:
            return await self._guarded_result(jp_text, user_msg, result)
        except Exception as exc:  # fail open: never drop an already-good line
            self.guard_failed_open += 1
            _log.warning(
                "polarity guard failed open (%s: %s); keeping the accepted "
                "translation (guard_failed_open=%d)",
                type(exc).__name__, exc, self.guard_failed_open,
            )
            return result

    async def _guarded_result(
        self, jp_text: str, user_msg: str, result: str
    ) -> str:
        """Compare JP/EN polarity; on a mismatch, make ONE extra completion that
        asks the model to fix it, and keep the retry only if the EN negation
        flips the required way. Never mutates JP; never more than one retry.

        Decision logic only -- callers reach it through _apply_polarity_guard,
        which owns the fail-open contract above."""
        # JP analysis is CPU-bound MeCab work, and the first call may lazily
        # build the fallback tagger (a potentially-heavy UniDic load); run both
        # off the event loop in one worker thread (same contract as process_pre
        # / build_ruby in translation_stage).
        pol = await asyncio.to_thread(self._analyze_jp_polarity, jp_text)
        en_neg = en_has_negation(result)

        direction: str | None = None
        if pol.negated and not en_neg:
            direction = "A"  # JP negated, EN dropped it
        elif (
            not pol.negated
            and pol.causative_grant
            and not pol.prohibitive
            and en_has_added_imperative_negation(result)
        ):
            # JP a causative/benefactive grant (死なせてやってくれ / 見せてくれ), EN
            # invented an imperative "Don't ...". Narrowed to this measured class
            # so natural -te requests (静かにして -> "Don't make noise.") no longer
            # trigger a wasted retry. prohibitive is redundant here (a grant tail
            # can't co-occur with a verb+な final) but kept as defense-in-depth.
            direction = "B"
        if direction is None:
            return result

        self.guard_fired += 1
        hint = _POLARITY_RETRY_HINT_A if direction == "A" else _POLARITY_RETRY_HINT_B
        _log.debug(
            "polarity guard fired (direction %s); one retry (guard_fired=%d)",
            direction, self.guard_fired,
        )
        # Bounded: the line already has a translation, so the retry gets one
        # 5 s request; a failure keeps the original (see the fail-open
        # wrapper in _apply_polarity_guard).
        retried = await self._complete(
            self._build_body(user_msg + "\n" + hint), timeout_s=5.0, attempts=1,
        )

        # Accept only if the retry FLIPPED the EN negative-token presence the
        # right way (A: now present; B: now absent) and is non-empty; otherwise
        # keep the original -- a failed retry must never worsen the output.
        accepted = bool(retried.strip()) and (
            en_has_negation(retried) if direction == "A"
            else not en_has_negation(retried)
        )
        if accepted:
            self.retry_accepted += 1
            _log.debug(
                "polarity guard retry accepted (direction %s, retry_accepted=%d)",
                direction, self.retry_accepted,
            )
            return retried
        _log.debug(
            "polarity guard retry rejected (direction %s); keeping original", direction,
        )
        return result

    async def translate(self, jp_text: str, genre: str, context: list[str],
                        series_context: str = "") -> str:
        hint = ""
        if self._pipeline is not None:
            # process_pre runs CPU-bound NER (fugashi + Sudachi) and glossary /
            # gazetteer lookups. Off the event loop so it can't stall the
            # other pipeline stages in the ~300ms hot path. Soft hint only;
            # the source is never mutated, so we send jp_text directly.
            hint = await asyncio.to_thread(self._pipeline.process_pre, jp_text)

        # series_context: the per-series domain hint composed by
        # gazetteer.series_profile.context_hint (translation.series_context_hint
        # plumbing). Its own prompt line, never merged into glossary_hints.
        user_msg = build_user_message(jp_text, genre, context, glossary_hints=hint,
                                      series_context=series_context)
        result = await self._complete(self._build_body(user_msg))

        if self._cfg.polarity_guard and jp_text:
            result = await self._apply_polarity_guard(jp_text, user_msg, result)
        return result

    async def sanity_check(self, genre: str = "variety") -> None:
        """Startup check: confirm no <think> tag appears in a fixed test response."""
        await self.translate("こんにちは。", genre, [])

    async def aclose(self) -> None:
        await self._client.aclose()


async def health_says_ready(client: httpx.AsyncClient, url: str) -> bool:
    """One readiness probe (plan phase 4, Decision 19): clears a translator
    problem, and tells a llama-server still loading its model from one that
    serves (Decision 21). llama-server serves ``/health`` 503 while it loads
    the model and 200 once it serves, while ``/v1/models`` answers during
    the load too. A server without ``/health`` (404/405, an external
    OpenAI-compatible endpoint) counts as ready: the caller only asks after
    ``/v1/models`` answered. Connection errors and timeouts are "not yet"."""
    try:
        response = await client.get(url, timeout=_READINESS_PROBE_TIMEOUT_S)
    except httpx.HTTPError:
        return False
    return response.status_code in (200, 404, 405)
