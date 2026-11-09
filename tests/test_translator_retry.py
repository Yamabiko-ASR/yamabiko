"""Translator retry behavior.

httpx.AsyncClient is swapped for a fake that records calls and can
raise transient / permanent errors on a per-call basis.
"""
from __future__ import annotations

import pytest
import httpx

from yamabiko.translation.translator import Translator, TranslationConfig


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: int = 0
        self.last_json = None  # request body of the most recent post()

    async def post(self, url, **kwargs):
        self.calls += 1
        self.last_json = kwargs.get("json")
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def aclose(self):
        pass


def _ok_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": text}}]},
        request=httpx.Request("POST", "http://fake"),
    )


def _cfg() -> TranslationConfig:
    return TranslationConfig(
        endpoint="http://localhost:8090/v1/chat/completions",
        model="test", max_tokens=40, temperature=0.1,
    )


@pytest.mark.asyncio
async def test_translate_succeeds_on_first_try():
    t = Translator(_cfg())
    t._client = _FakeClient([_ok_response("hello")])
    result = await t.translate("こんにちは", "variety", [])
    assert result == "hello"
    assert t._client.calls == 1


@pytest.mark.asyncio
async def test_translate_sends_enable_thinking_false_in_request_body():
    """The outgoing request body must carry
    chat_template_kwargs={"enable_thinking": False} so llama-server's jinja
    engine switches Qwen3 to non-thinking mode. Nothing else verified this
    (the retry test's fake client discarded the body), so a refactor that
    dropped the field would ship green (issue #60, rule #3 gap).

    Seeded defect: delete the chat_template_kwargs line in translator.py
    -> the body has no such key -> this fails.
    """
    t = Translator(_cfg())
    t._client = _FakeClient([_ok_response("hi")])
    await t.translate("こんにちは", "variety", [])
    body = t._client.last_json
    assert body is not None, "translate() sent no JSON body"
    assert body.get("chat_template_kwargs") == {"enable_thinking": False}, (
        f"request body missing enable_thinking=False; got "
        f"chat_template_kwargs={body.get('chat_template_kwargs')!r}"
    )


@pytest.mark.asyncio
async def test_translate_raises_when_thinking_leaks_into_reasoning_content():
    """Under --reasoning-format auto (or an external endpoint we don't control),
    a leaked Qwen3 thinking block lands in message.reasoning_content, NOT
    message.content -- so the legacy `<think>`-in-content guard never fires and
    thinking silently corrupts/empties the subtitle. A non-empty
    reasoning_content must raise the same thinking-detected error (issue #60).

    Seeded defect: drop the reasoning_content check in translator.py -> a
    response carrying reasoning_content returns silently instead of raising
    -> this fails.
    """
    leaked = httpx.Response(
        200,
        json={"choices": [{"message": {
            "content": "",
            "reasoning_content": "<think>the user said hello</think>",
        }}]},
        request=httpx.Request("POST", "http://fake"),
    )
    t = Translator(_cfg())
    t._client = _FakeClient([leaked])
    with pytest.raises(RuntimeError, match="[Tt]hinking"):
        await t.translate("こんにちは", "variety", [])


@pytest.mark.asyncio
async def test_translate_handles_null_content():
    """llama.cpp's OpenAI server can return content: null (a truncated /
    empty completion). translate() must return '' (-> dropped downstream),
    not crash with TypeError on the '<think>' substring check (which is fatal
    at startup sanity_check -> generic exit-43 dialog).

    Seeded defect: revert `... ["content"] or ""` to `... ["content"]`
    -> `'<think>' in None` raises TypeError -> this fails.
    """
    t = Translator(_cfg())
    null_resp = httpx.Response(
        200,
        json={"choices": [{"message": {"content": None}}]},
        request=httpx.Request("POST", "http://fake"),
    )
    t._client = _FakeClient([null_resp])
    result = await t.translate("こんにちは", "variety", [])
    assert result == ""


@pytest.mark.asyncio
async def test_process_pre_runs_off_the_event_loop():
    """process_pre (CPU-bound NER + synchronous glossary JSON writes) must
    run OFF the event-loop thread so it doesn't stall the other pipeline
    stages in the ~300ms hot path.

    Seeded defect: drop the asyncio.to_thread wrapper around process_pre
    -> it runs on the loop thread -> recorded ident == loop ident -> fails.
    """
    import threading

    loop_ident = threading.get_ident()

    class _ThreadRecordingPipeline:
        def __init__(self):
            self.pre_ident = None

        def process_pre(self, jp):
            self.pre_ident = threading.get_ident()
            return ""

    pipe = _ThreadRecordingPipeline()
    t = Translator(_cfg(), pipeline=pipe)
    t._client = _FakeClient([_ok_response("hello")])
    await t.translate("こんにちは", "variety", [])
    assert pipe.pre_ident not in (None, loop_ident), (
        "process_pre ran on the event-loop thread"
    )


@pytest.mark.asyncio
async def test_translate_puts_raw_jp_and_gazetteer_hint_into_user_message():
    """process_pre returns the hint string. translate() must send its RAW
    jp_text arg unchanged AND route the soft hint into build_user_message's
    glossary_hints, so both appear in the outgoing user message.

    Seeded defect: pass glossary_hints="" (drop the hint) -> 'Rimuru' absent
    from the body -> red. Also red if translate() sends the process_pre
    return in place of jp_text -> the raw Japanese source is lost from the
    prompt.
    """

    class _HintPipeline:
        def process_pre(self, jp):
            return "Use these exact English names for the proper nouns: リムル=Rimuru."

    t = Translator(_cfg(), pipeline=_HintPipeline())
    t._client = _FakeClient([_ok_response("Rimuru arrived.")])
    await t.translate("リムルさんが来た。", "anime", [])
    body = t._client.last_json
    assert body is not None, "translate() sent no JSON body"
    user_msg = body["messages"][1]["content"]
    assert "リムルさんが来た。" in user_msg, (
        f"raw Japanese source must be sent unchanged; got user message {user_msg!r}"
    )
    assert "Rimuru" in user_msg, (
        f"the soft gazetteer hint must reach the prompt via glossary_hints; "
        f"got user message {user_msg!r}"
    )


@pytest.mark.asyncio
async def test_translate_retries_once_on_connect_error():
    t = Translator(_cfg())
    t._client = _FakeClient([
        httpx.ConnectError("connection refused"),
        _ok_response("world"),
    ])
    result = await t.translate("こんにちは", "variety", [])
    assert result == "world"
    assert t._client.calls == 2


@pytest.mark.asyncio
async def test_translate_retries_once_on_timeout():
    t = Translator(_cfg())
    t._client = _FakeClient([
        httpx.ReadTimeout("read timeout"),
        _ok_response("timed-then-ok"),
    ])
    result = await t.translate("こんにちは", "variety", [])
    assert result == "timed-then-ok"
    assert t._client.calls == 2


@pytest.mark.asyncio
async def test_translate_gives_up_after_second_network_error():
    t = Translator(_cfg())
    t._client = _FakeClient([
        httpx.ConnectError("first refusal"),
        httpx.ConnectError("second refusal"),
    ])
    with pytest.raises(httpx.ConnectError):
        await t.translate("こんにちは", "variety", [])
    assert t._client.calls == 2


@pytest.mark.asyncio
async def test_translate_does_not_retry_on_4xx():
    err_response = httpx.Response(
        400, json={"error": "bad request"},
        request=httpx.Request("POST", "http://fake"),
    )
    t = Translator(_cfg())
    t._client = _FakeClient([err_response])
    with pytest.raises(httpx.HTTPStatusError):
        await t.translate("こんにちは", "variety", [])
    assert t._client.calls == 1


# --- _TRANSIENT_ERRORS must cover ConnectTimeout (TCP connect stalls while
# llama-server is loading the model), PoolTimeout (pool exhausted under
# GPU-load latency), and RemoteProtocolError (server closes the connection
# mid-response during a restart) in addition to the basic leaf classes.
# Each of these escapes the retry loop on attempt 0 if omitted, turning a
# routine blip into a dropped subtitle.
# Seeded defect: revert _TRANSIENT_ERRORS to only the four basic leaf
# classes -> all three parametrized cases go red (exception propagates,
# calls == 1).

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectTimeout("connect timed out"),
        httpx.PoolTimeout("pool timed out"),
        httpx.RemoteProtocolError("server disconnected without response"),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_translate_retries_once_on_transient_subtype(exc):
    t = Translator(_cfg())
    t._client = _FakeClient([exc, _ok_response("recovered")])
    result = await t.translate("こんにちは", "variety", [])
    assert result == "recovered"
    assert t._client.calls == 2


@pytest.mark.asyncio
async def test_translate_does_not_retry_on_local_protocol_error():
    """httpx.LocalProtocolError means WE built a malformed request --
    retrying cannot fix a client-side bug, so it must propagate on
    attempt 0. Pins the deliberate choice of (TimeoutException,
    NetworkError, RemoteProtocolError) over the broader
    httpx.TransportError, which would swallow client bugs into the
    retry path.

    Seeded defect: widen _TRANSIENT_ERRORS to (httpx.TransportError,)
    -> the error is retried (calls == 2) -> red.
    """
    t = Translator(_cfg())
    t._client = _FakeClient([
        httpx.LocalProtocolError("malformed request"),
        _ok_response("should never be reached"),
    ])
    with pytest.raises(httpx.LocalProtocolError):
        await t.translate("こんにちは", "variety", [])
    assert t._client.calls == 1


@pytest.mark.asyncio
async def test_translate_retries_once_on_5xx():
    err_response = httpx.Response(
        503, json={"error": "overloaded"},
        request=httpx.Request("POST", "http://fake"),
    )
    t = Translator(_cfg())
    t._client = _FakeClient([err_response, _ok_response("recovered")])
    result = await t.translate("こんにちは", "variety", [])
    assert result == "recovered"
    assert t._client.calls == 2
