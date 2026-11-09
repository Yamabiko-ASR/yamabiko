"""Smoke-test page tests — the llama-server probe is injected.

Production uses httpx against /health + /v1/chat/completions; tests use
a fake that returns canned responses or raises to simulate failure.

Probe calls run on a worker QThread, so results arrive via queued signals
-- tests wait with qtbot.waitUntil instead of asserting synchronously
after run_smoke().
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("pytestqt")


class FakeProbe:
    def __init__(self, *, health_ok: bool = True, translate_result: str = "Hello"):
        self.health_ok = health_ok
        self.translate_result = translate_result
        self.calls: list[str] = []
        self.call_thread_ids: set[int] = set()

    def check_health(self) -> bool:
        self.calls.append("health")
        self.call_thread_ids.add(threading.get_ident())
        return self.health_ok

    def translate(self, text: str) -> str:
        self.calls.append(f"translate:{text}")
        self.call_thread_ids.add(threading.get_ident())
        if not self.health_ok:
            raise RuntimeError("not healthy")
        return self.translate_result


def _failure_surfaced(page) -> bool:
    return not page.alert_box.isHidden()


@pytest.fixture
def make_page(qtbot):
    """SmokePage factory whose teardown stops the probe thread.

    The page's worker QThread idles between probe calls; in production
    the wizard's release_background_threads() stops it before page
    destruction. pytest-qt destroys widgets without that hook, and
    destroying a page whose child QThread still runs is a Qt 6 qFatal
    process abort — so every page built here gets shutdown_thread() at
    test end.
    """
    from yamabiko.wizard_pages.smoke import SmokePage

    pages = []

    def _make(**kwargs):
        page = SmokePage(**kwargs)
        qtbot.addWidget(page)
        pages.append(page)
        return page

    yield _make
    for page in pages:
        page.shutdown_thread(wait_ms=5000)


def test_smoke_success(qtbot, make_page):
    probe = FakeProbe(health_ok=True, translate_result="Hello")
    page = make_page(probe=probe)
    page.run_smoke()
    qtbot.waitUntil(page.smoke_ok, timeout=5000)
    assert page.rows.row_text(1) == "Test translation, こんにちは, →, Hello"
    assert page.rows.row_state(1) == "done"
    assert page.alert_box.isHidden(), "Retry and Skip live only in the failure alert (L7)"


def test_empty_translation_response_cannot_mark_smoke_ready(qtbot, make_page):
    """A 200 response with no English text is a failed translation probe.

    Seeded defect: accept the probe's empty string as a successful result ->
    smoke_ok becomes true and this test goes red.
    """
    page = make_page(probe=FakeProbe(health_ok=True, translate_result=""))
    page.run_smoke()
    qtbot.waitUntil(
        lambda: page.smoke_ok() or _failure_surfaced(page), timeout=5000,
    )

    assert page.smoke_ok() is False
    assert page.alert.message() == "The test translation failed."


def test_probe_calls_run_off_the_gui_thread(qtbot, make_page):
    """check_health and translate execute on a worker thread -- a synchronous
    httpx.post(timeout=30) on the GUI thread freezes the wizard into Windows'
    'Not Responding' ghost state during the translation test.

    Seeded defect: call self._probe.translate(...) inline on the page instead
    of via the worker → the recorded thread id equals the GUI thread's → red.
    """
    probe = FakeProbe(health_ok=True, translate_result="Hello")
    page = make_page(probe=probe)
    gui_thread_id = threading.get_ident()
    page.run_smoke()
    qtbot.waitUntil(page.smoke_ok, timeout=5000)
    assert probe.call_thread_ids, "probe never called"
    assert gui_thread_id not in probe.call_thread_ids, (
        "probe calls ran on the GUI thread; they must execute on a worker thread"
    )


def test_smoke_health_fail(qtbot, make_page):
    probe = FakeProbe(health_ok=False)
    # max_health_attempts=1 -> a single refusal exhausts the budget and the
    # failure is surfaced on the first health result.
    page = make_page(probe=probe, max_health_attempts=1)
    page.run_smoke()
    qtbot.waitUntil(lambda: _failure_surfaced(page), timeout=5000)
    assert page.smoke_ok() is False


def test_no_answer_shows_the_alert_with_retry_skip_and_the_skip_note(qtbot, make_page):
    """L7: when nothing answers on the translator's port (connection refused
    on every attempt), the alert says so in the spec's words, Retry
    (primary) and Skip live in it with the muted skip note, the start row is
    paused (Decision 131), and Next stays disabled.

    Seeded defect: keep the start row "now" on the exhausted budget, or
    drop the skip note -> red.
    """
    page = make_page(probe=FakeProbe(health_ok=False), max_health_attempts=1)
    page.run_smoke()
    qtbot.waitUntil(lambda: _failure_surfaced(page), timeout=5000)

    assert page.alert.message() == "The translator didn't answer yet."
    assert page.alert.help_text() == "The first start can take a minute while the model loads."
    assert page.alert.after_text() == "If you skip, Yamabiko tries again when it starts."
    assert page.retry_button.property("primary") is True
    assert page.skip_button.isVisibleTo(page)  # inside the shown alert (Decision 148)
    assert page.rows.row_state(0) == "paused"
    assert not page.isComplete()


def test_health_poll_does_not_fail_on_first_refusal(qtbot, make_page):
    """A single connection-refused (translator still starting) keeps the
    "Starting the translator" row running and the poll going -- NOT the
    failure alert.

    Seeded-defect drill: revert to a single check-then-fail -> the first
    refusal shows the alert -> red.
    """
    probe = FakeProbe(health_ok=False)
    page = make_page(probe=probe, max_health_attempts=5, health_poll_interval_ms=10_000)
    page._ensure_worker()
    # The page's slot is connected first, so it has handled the refusal by
    # the time the blocker sees the signal.
    with qtbot.waitSignal(page._worker.health_result, timeout=5000):
        page.run_smoke()

    assert page.alert_box.isHidden(), "the first refusal showed the failure alert"
    assert page.rows.row_state(0) == "now"
    assert page.smoke_ok() is False
    assert probe.calls.count("health") == 1  # exactly one attempt so far


def test_health_poll_succeeds_after_initial_refusal(qtbot, make_page):
    """The poll recovers: once health comes up on a later attempt, the
    translation test runs and the page completes."""

    class _LaterHealthyProbe:
        def __init__(self, fail_times: int):
            self._fail_times = fail_times
            self.n = 0

        def check_health(self) -> bool:
            self.n += 1
            return self.n > self._fail_times

        def translate(self, text: str) -> str:
            return "Hello"

    probe = _LaterHealthyProbe(fail_times=1)
    # Short interval so attempt 2 fires within the test window.
    page = make_page(probe=probe, max_health_attempts=5, health_poll_interval_ms=10)
    page.run_smoke()
    qtbot.waitUntil(page.smoke_ok, timeout=5000)
    assert page.rows.row_text(1).endswith("Hello")
    assert probe.n >= 2


def test_skip_completes_the_step_and_asks_to_go_on(qtbot, make_page):
    """L7: Skip (in the failure alert) marks the step complete and asks the
    wizard to move on to Done (the shell connects ``skipped`` to next).

    Seeded defect: never emit ``skipped`` -> red.
    """
    probe = FakeProbe(health_ok=False)
    page = make_page(probe=probe, max_health_attempts=1)
    asked: list = []
    page.skipped.connect(lambda: asked.append(True))
    page.run_smoke()
    qtbot.waitUntil(lambda: _failure_surfaced(page), timeout=5000)

    page.skip_button.click()

    assert page.isComplete()
    assert asked == [True]


def test_retry_after_skip_clears_skip_state(qtbot, make_page):
    # First time: fail → user clicks Skip → wizard completes via skip path.
    probe = FakeProbe(health_ok=False)
    page = make_page(probe=probe, max_health_attempts=1)
    page.run_smoke()
    qtbot.waitUntil(lambda: _failure_surfaced(page), timeout=5000)
    page.skip_button.click()
    assert page.isComplete()
    assert page.smoke_ok() is False

    # Now flip the probe to healthy and retry. _skipped should reset.
    probe.health_ok = True
    page.run_smoke()
    qtbot.waitUntil(page.smoke_ok, timeout=5000)
    assert page._skipped is False  # white-box check; see Fix 3
    assert page.alert_box.isHidden(), "a passed Retry left the failure alert up"


def test_translate_exception_path(qtbot, make_page):
    class TranslateFailProbe:
        def check_health(self) -> bool:
            return True

        def translate(self, text: str) -> str:
            raise RuntimeError("llm blew up")

    page = make_page(probe=TranslateFailProbe())
    page.run_smoke()
    qtbot.waitUntil(lambda: _failure_surfaced(page), timeout=5000)
    assert page.alert.message() == "The test translation failed."
    assert page.alert.help_text() == "llm blew up"
    assert page.rows.row_state(0) == "done"
    assert page.rows.row_state(1) == "paused"
    assert page.smoke_ok() is False


def test_retry_mid_flight_discards_stale_health_result(qtbot, make_page):
    """A Retry while a health probe is in flight must discard the stale
    result — otherwise the page reacts to BOTH results and runs the
    translation test twice.

    Seeded defect: drop the generation check in the health-result handler →
    two translate calls → red.
    """
    gate = threading.Event()

    class GatedProbe:
        def __init__(self) -> None:
            self.translate_calls = 0

        def check_health(self) -> bool:
            gate.wait(timeout=10.0)
            return True

        def translate(self, text: str) -> str:
            self.translate_calls += 1
            return "Hello"

    probe = GatedProbe()
    page = make_page(probe=probe)
    page.run_smoke()   # generation 1 — health probe blocks on the gate
    page.run_smoke()   # generation 2 — user clicked Retry mid-flight
    gate.set()
    qtbot.waitUntil(page.smoke_ok, timeout=5000)
    qtbot.wait(200)    # allow any (wrong) duplicate translate to land
    assert probe.translate_calls == 1


def test_going_back_mid_poll_stops_the_health_checks_until_the_step_is_entered_again(
    qtbot, make_page,
):
    """Back from the Translator step while the translator is still starting
    stops the /health poll at once (it ran on, hidden, for up to 30 s); a
    request already sent may still land, its result discarded. Entering the
    step again starts a fresh check on the same worker thread.

    Seeded defect: drop SmokePage.cleanupPage -> the poll keeps calling
    /health on the hidden step -> red.
    """
    probe = FakeProbe(health_ok=False)
    page = make_page(probe=probe, max_health_attempts=1000, health_poll_interval_ms=10)
    page.initializePage()
    qtbot.waitUntil(lambda: probe.calls.count("health") >= 2, timeout=5000)
    thread = page._thread

    page.cleanupPage()
    after_back = probe.calls.count("health")
    qtbot.wait(300)
    assert probe.calls.count("health") <= after_back + 1, "the poll ran on after Back"

    page.initializePage()
    qtbot.waitUntil(lambda: probe.calls.count("health") >= after_back + 3, timeout=5000)
    assert page._thread is thread


def test_shutdown_thread_detaches_inflight_probe_thread_into_keeper(qtbot, make_page):
    """Closing the wizard while a probe call is mid-flight must detach the
    worker thread (no Qt parent, parked in thread_keeper) -- never destroy or
    indefinitely join it. The detached thread exits on its own once the httpx
    timeout fires.

    Seeded defect: remove the thread_keeper.release call → red.
    """
    from yamabiko.ui import thread_keeper

    gate = threading.Event()
    entered = threading.Event()

    class BlockedProbe:
        def check_health(self) -> bool:
            entered.set()
            gate.wait(timeout=10.0)
            return False

        def translate(self, text: str) -> str:
            return "Hello"

    page = make_page(probe=BlockedProbe(), max_health_attempts=1)
    page.run_smoke()
    qtbot.waitUntil(entered.is_set, timeout=5000)
    thread = page._thread
    assert thread is not None
    try:
        page.shutdown_thread(wait_ms=100)
        assert thread.isRunning(), "blocked probe thread should still be draining"
        assert thread.parent() is None
        assert any(t is thread for t, _ in thread_keeper._LINGERING)
        assert page._thread is None
    finally:
        gate.set()
        thread.wait(5000)
        thread_keeper.discard(thread)


# ---------------------------------------------------------------------------
# P1: the production probe's endpoint must come from the MERGED config at
# page-entry time. The watchdog honors translation.endpoint overrides and
# installer/README.md documents seeding user_overrides.yaml BEFORE first
# run; a probe hardcoded to localhost:8090 showed a guaranteed false smoke
# failure to every endpoint-override user forced through the upgrade wizard.
# The probe is an injectable seam by design, so the recording stand-in below
# is fixture mocking around the page, not subject-mocking.
# ---------------------------------------------------------------------------


def _install_recording_http_probe(monkeypatch) -> list:
    """Replace smoke._HTTPLlamaProbe with a healthy recording probe; returns
    the list that collects each constructed probe's endpoint."""
    from yamabiko.wizard_pages import smoke

    endpoints: list = []

    class _Recording:
        def __init__(self, endpoint=None, *, model=None):
            endpoints.append(endpoint)

        def check_health(self) -> bool:
            return True

        def translate(self, text: str) -> str:
            return "Hello"

    monkeypatch.setattr(smoke, "_HTTPLlamaProbe", _Recording)
    return endpoints


def test_initialize_page_probes_the_config_override_endpoint(
    qtbot, make_page, monkeypatch,
):
    """With a translation.endpoint override in the merged config, entering
    the page must build the production probe against THAT endpoint and run
    the smoke check through it to completion; the first row names that
    endpoint's host:port.

    Seeded defect: revert to the construction-time `_HTTPLlamaProbe()`
    default (or drop the _resolve_config_endpoint call in initializePage)
    -> the recorded endpoint is the localhost default -> red.
    """
    override = "http://gpubox.lan:9191/v1/chat/completions"
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"translation": {"endpoint": override}},
    )
    endpoints = _install_recording_http_probe(monkeypatch)

    page = make_page()  # no injected probe: the production wizard path
    page.initializePage()
    qtbot.waitUntil(page.smoke_ok, timeout=5000)

    assert endpoints == [override], (
        f"smoke probe was not built against the merged-config endpoint "
        f"override; got {endpoints!r}"
    )
    assert page.rows.row_text(0) == "Starting the translator, gpubox.lan:9191"
    assert page.rows.row_text(1).endswith("Hello")


def test_initialize_page_falls_back_to_default_endpoint_when_config_load_fails(
    qtbot, make_page, monkeypatch,
):
    """A broken user_overrides.yaml (config load raises) must degrade to the
    shipped-default endpoint, not crash the wizard or probe nothing.

    Seeded defect: let the config exception propagate out of
    _resolve_config_endpoint -> initializePage raises -> red.
    """
    def _boom():
        raise OSError("user_overrides.yaml unreadable")

    monkeypatch.setattr("yamabiko.config.load_full_config", _boom)
    endpoints = _install_recording_http_probe(monkeypatch)

    page = make_page()
    page.initializePage()
    qtbot.waitUntil(page.smoke_ok, timeout=5000)

    from yamabiko.wizard_pages.smoke import _DEFAULT_ENDPOINT
    assert endpoints == [_DEFAULT_ENDPOINT]
    assert endpoints == ["http://localhost:8090/v1/chat/completions"]


def test_initialize_page_keeps_an_injected_probe(qtbot, make_page, monkeypatch):
    """The constructor's injectable probe seam must survive initializePage
    -- tests (and the wizard-walk test, which assigns ``_probe`` directly)
    rely on the page never rebuilding a probe it was given.

    Seeded defect: rebuild the production probe unconditionally in
    initializePage -> the exploding factory fires -> red.
    """
    from yamabiko.wizard_pages import smoke

    def _explode(*args, **kwargs):
        raise AssertionError(
            "initializePage rebuilt the probe over an injected one"
        )

    monkeypatch.setattr(smoke, "_HTTPLlamaProbe", _explode)
    probe = FakeProbe(health_ok=True, translate_result="Hello")
    page = make_page(probe=probe)
    page.initializePage()
    qtbot.waitUntil(page.smoke_ok, timeout=5000)
    assert page.rows.row_text(1).endswith("Hello")
    assert probe.calls, "the injected probe was never exercised"


# ---------------------------------------------------------------------------
# Override-endpoint parity with the runtime translator. Two residual
# false-failure paths hit exactly the population the endpoint-override fix
# targets: (a) the translate POST hardcoded model=TRANSLATOR_ALIAS while the
# runtime sends merged translation.model (external vLLM/Ollama/OpenRouter
# 404s the alias); (b) check_health hard-required /health == 200 while the
# runtime's wait_until_ready tolerates gated/absent /health on external
# endpoints (only 5xx means "still loading"). These drive the REAL
# _HTTPLlamaProbe through a fake httpx module (sys.modules seam), so the
# request URL, body, and status handling are the production code's own.
# ---------------------------------------------------------------------------


def _install_fake_httpx(monkeypatch, *, health_status=200,
                        translate_reply="Hello", reply_model=None, reply_payload=None):
    """Swap sys.modules['httpx'] for a recorder. Returns a namespace with
    ``gets`` (urls) and ``posts`` ((url, json_body) tuples). ``reply_payload``
    replaces the whole chat-completion reply; a str is a body that isn't
    JSON (an HTML error page)."""
    import json as jsonlib
    import sys
    import types

    records = types.SimpleNamespace(gets=[], posts=[])

    class _Resp:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self.text = payload if isinstance(payload, str) else jsonlib.dumps(payload)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return jsonlib.loads(self.text)  # as httpx's: ValueError on a body that isn't JSON

    def _get(url, timeout=None, trust_env=True):
        records.gets.append(url)
        return _Resp(health_status)

    def _post(url, json=None, timeout=None, trust_env=True):
        records.posts.append((url, json))
        payload = {"choices": [{"message": {"content": translate_reply}}]}
        if reply_model is not None:
            payload["model"] = reply_model
        if reply_payload is not None:
            payload = reply_payload
        return _Resp(200, payload)

    monkeypatch.setitem(
        sys.modules, "httpx", types.SimpleNamespace(get=_get, post=_post),
    )
    return records


def test_smoke_post_disables_qwen_thinking_like_runtime_translator(monkeypatch):
    """The wizard probe must request Qwen's non-thinking chat template.

    Without this option Qwen3.5 can spend the entire 40-token smoke budget in
    ``reasoning_content`` and return an empty ``content`` field.

    Seeded defect: remove ``chat_template_kwargs`` from the POST body -> the
    recorded request no longer disables thinking and this test goes red.
    """
    records = _install_fake_httpx(monkeypatch, translate_reply="Hello")

    from yamabiko.wizard_pages.smoke import _HTTPLlamaProbe

    probe = _HTTPLlamaProbe()
    assert probe.translate("こんにちは") == "Hello"

    assert records.posts, "translate POST never fired"
    _url, body = records.posts[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_smoke_reports_another_service_answering_with_a_different_model(
    qtbot, make_page, monkeypatch,
):
    """A different local server (LM Studio, a second llama-server) already
    on the endpoint answers the chat completion with its own model name.
    The smoke test must not pass on that reply: it names the host:port and
    the foreign model and points at Settings' translator endpoint, and the
    test row pauses (Decision 89).

    Seeded defects: drop the reply-model check in _HTTPLlamaProbe.translate
    -> the foreign reply marks the translator ready -> red; leave the test
    row running -> red.
    """
    records = _install_fake_httpx(
        monkeypatch, health_status=200, translate_reply="Hello",
        reply_model="lmstudio-community/gemma-3-4b",
    )

    page = make_page()  # production probe: merged-config endpoint + alias
    page.initializePage()
    qtbot.waitUntil(lambda: _failure_surfaced(page), timeout=5000)

    assert records.posts, "translate POST never fired"
    assert page.alert.message() == (
        'Another app is answering on localhost:8090 (model "lmstudio-community/gemma-3-4b").'
    )
    assert page.alert.help_text() == (
        "Close it, then click Retry. Or skip and set the translator endpoint in Settings."
    )
    assert page.rows.row_state(1) == "paused"
    assert page.smoke_ok() is False


@pytest.mark.parametrize(("reply", "help_line", "logged"), [
    ({"choices": [{"message": {"content": "  "}}]},
     "The translator sent an empty answer.",
     "TranslatorReplyError: The translator sent an empty answer."),
    ({"error": "busy"},
     "The translator sent an answer Yamabiko couldn't read.",
     "KeyError: 'choices'"),
    ("<html><body>502 Bad Gateway</body></html>",
     "The translator sent an answer Yamabiko couldn't read.",
     "JSONDecodeError: Expecting value"),
])
def test_an_empty_or_unreadable_reply_gets_a_plain_help_line(
    qtbot, make_page, monkeypatch, caplog, reply, help_line, logged,
):
    """The spec's plain-copy rule: a reply with no text, one that isn't the
    chat-completion JSON, or one that isn't JSON at all (a proxy's HTML
    page) says so in a plain sentence under "The test translation failed."
    (not "translator returned no English text" or a KeyError's "'choices'");
    the log keeps the full error and the reply itself, so a report says what
    answered (the server's "busy", the HTML page).

    Seeded defects: show the parse error's own text, or the old empty-reply
    wording -> red; drop the log line, or log it without the reply -> red;
    raise the not-JSON error with the parsed ``data`` instead of ``r.text``
    -> an UnboundLocalError's text in the alert -> red.
    """
    _install_fake_httpx(monkeypatch, health_status=200, reply_payload=reply)

    page = make_page()  # production probe against the fake endpoint
    page.initializePage()
    qtbot.waitUntil(lambda: _failure_surfaced(page), timeout=5000)

    assert page.alert.message() == "The test translation failed."
    assert page.alert.help_text() == help_line
    assert logged in caplog.text
    assert f"reply={reply!r}" in caplog.text


def test_smoke_post_body_carries_merged_config_model_on_override_endpoint(
    qtbot, make_page, monkeypatch,
):
    """With translation.endpoint AND translation.model overridden, the
    translate POST must address the override model -- the runtime sends
    merged translation.model, and an external server 404s the bundled
    alias, turning a working endpoint into a false 'Translation test
    failed'.

    Seeded defect: revert the POST body (or the probe construction) to
    TRANSLATOR_ALIAS -> the recorded body carries the alias -> red.
    """
    override = "http://gpubox.lan:9191/v1/chat/completions"
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"translation": {"endpoint": override,
                                 "model": "qwen-custom-vllm"}},
    )
    records = _install_fake_httpx(monkeypatch, health_status=200)

    page = make_page()  # no injected probe: the production wizard path
    page.initializePage()
    qtbot.waitUntil(page.smoke_ok, timeout=5000)

    assert records.posts, "translate POST never fired"
    url, body = records.posts[0]
    assert url == override
    assert body["model"] == "qwen-custom-vllm", (
        f"POST body must carry the merged translation.model override; "
        f"got {body['model']!r}"
    )


def test_override_endpoint_health_404_still_reaches_translate_and_passes(
    qtbot, make_page, monkeypatch,
):
    """An override endpoint that gates or lacks /health (LM Studio-class,
    path-prefixed proxies) but serves chat completions must pass the smoke
    test -- mirrors the runtime's wait_until_ready, where a completed
    non-5xx /health response on an external endpoint skips the gate and
    the translate attempt arbitrates.

    Seeded defect: revert check_health to the strict status==200 gate for
    every endpoint -> the poll budget exhausts, translate never runs ->
    waitUntil times out -> red.
    """
    override = "http://gpubox.lan:9191/v1/chat/completions"
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"translation": {"endpoint": override}},
    )
    records = _install_fake_httpx(monkeypatch, health_status=404)

    page = make_page(max_health_attempts=1)
    page.initializePage()
    qtbot.waitUntil(page.smoke_ok, timeout=5000)

    assert records.posts, "translate POST never fired despite alive endpoint"
    assert page.rows.row_text(1).endswith("Hello")


def test_override_endpoint_health_5xx_keeps_polling_as_still_loading(
    qtbot, make_page, monkeypatch,
):
    """5xx from an override endpoint's /health is a still-loading signal
    (llama itself serves 503 while loading; gateways return 502/504 in
    front of a loading backend) -- the poll continues and translate is NOT
    attempted, exactly like the runtime's wait_until_ready.

    Seeded defect: treat ANY completed /health response as alive on
    override endpoints -> translate fires on the 503 -> red.
    """
    override = "http://gpubox.lan:9191/v1/chat/completions"
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"translation": {"endpoint": override}},
    )
    records = _install_fake_httpx(monkeypatch, health_status=503)

    page = make_page(max_health_attempts=1)
    page.initializePage()
    qtbot.waitUntil(lambda: _failure_surfaced(page), timeout=5000)

    assert records.posts == [], (
        "translate must not run while the endpoint reports still-loading"
    )
    assert page.smoke_ok() is False


def test_bundled_default_endpoint_keeps_strict_health_200_gate(
    qtbot, make_page, monkeypatch,
):
    """The bundled default endpoint owns its llama-server, where
    /health == 200 is meaningful -- a non-200 (here 404) must NOT be
    treated as alive, and translate must not be attempted.

    Seeded defect: extend the external-endpoint tolerance to the bundled
    default -> the 404 proceeds to translate and the smoke passes -> red.
    """
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {},  # no override: resolves to the shipped default
    )
    records = _install_fake_httpx(monkeypatch, health_status=404)

    page = make_page(max_health_attempts=1)
    page.initializePage()
    qtbot.waitUntil(lambda: _failure_surfaced(page), timeout=5000)

    from yamabiko.wizard_pages.smoke import _DEFAULT_ENDPOINT
    assert records.gets, "health probe never fired"
    assert records.gets[0].startswith("http://localhost:8090"), (
        f"expected the bundled default host, got {records.gets[0]!r} "
        f"(default endpoint: {_DEFAULT_ENDPOINT})"
    )
    assert records.posts == [], (
        "translate must not run when the bundled llama-server's /health "
        "is not 200"
    )
    assert page.smoke_ok() is False


def test_a_translator_that_answers_while_loading_has_started(qtbot, make_page, monkeypatch):
    """Decision 131 (R2): a translator that answers HTTP at all (the bundled
    llama-server serves /health 503 while it loads the model) has started:
    "Starting the translator" is checked, and when it never gets ready
    "Test translation" is the paused row under "The translator didn't
    answer yet." (the mockup's picture). Only an endpoint that never
    answered pauses the start row
    (test_no_answer_shows_the_alert_with_retry_skip_and_the_skip_note).

    Seeded defect: ignore the answered flag (pause the start row whenever
    the budget runs out) -> red.
    """
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {},  # no override: the bundled default endpoint
    )
    records = _install_fake_httpx(monkeypatch, health_status=503)

    page = make_page(max_health_attempts=1)
    page.initializePage()
    qtbot.waitUntil(lambda: _failure_surfaced(page), timeout=5000)

    assert records.gets and records.posts == []
    assert page.alert.message() == "The translator didn't answer yet."
    assert page.rows.row_state(0) == "done"
    assert page.rows.row_state(1) == "paused"
    assert not page.isComplete()


def test_shutdown_thread_joins_idle_probe_thread_without_detach(qtbot, make_page):
    """With no probe call in flight, shutdown_thread() joins the worker thread
    and adds nothing to thread_keeper (no leak on clean close).

    Seeded defect: release unconditionally → registry gains an entry → red.
    """
    from yamabiko.ui import thread_keeper

    probe = FakeProbe(health_ok=True)
    page = make_page(probe=probe)
    page.run_smoke()
    qtbot.waitUntil(page.smoke_ok, timeout=5000)  # worker idle afterwards

    before = len(thread_keeper._LINGERING)
    page.shutdown_thread(wait_ms=2000)
    assert len(thread_keeper._LINGERING) == before
    assert page._thread is None
