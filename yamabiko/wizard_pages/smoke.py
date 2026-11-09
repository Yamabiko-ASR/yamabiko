"""Wizard step "Translator": a quick check that the translator answers.

Probes `/health` then fires one translation request ("こんにちは"). The
probe is abstracted so tests can swap it out. Two rows show the progress
("Starting the translator" with the endpoint's host:port, "Test
translation" with the reply); a failure shows an alert holding Retry and
Skip. Failure is non-blocking: Skip goes on to Done, and Yamabiko tries
again when it starts.

Threading: the httpx calls block (check_health up to 5 s, translate up to
30 s on a cold model), so they run on a persistent worker QThread, never on
the Qt GUI thread (which Windows would ghost as 'Not Responding'). The poll
budget, retry timer, and Retry/Skip buttons stay on the main thread;
requests and results cross the thread boundary as queued signals carrying a
generation counter so a Retry mid-flight discards the superseded result.
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Optional, Protocol

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QVBoxLayout, QWidget, QWizardPage

from yamabiko import loopback_http
from yamabiko.models import TRANSLATOR_ALIAS
from yamabiko.overlay_app.controls import ui_font
from yamabiko.overlay_app.theme import MIST
from yamabiko.ui import thread_keeper
from yamabiko.ui.error_report import redact_user
from yamabiko.ui.error_window import mono_font
from yamabiko.wizard_pages.parts import Alert, Run, StatusRows

_log = logging.getLogger(__name__)

SAMPLE_JP = "こんにちは"
NO_ANSWER = "The translator didn't answer yet."
NO_ANSWER_HELP = "The first start can take a minute while the model loads."
TEST_FAILED = "The test translation failed."
MISMATCH_HELP = "Close it, then click Retry. Or skip and set the translator endpoint in Settings."
SKIP_NOTE = "If you skip, Yamabiko tries again when it starts."
EMPTY_REPLY = "The translator sent an empty answer."
UNREADABLE_REPLY = "The translator sent an answer Yamabiko couldn't read."
_ROW_NAMES = ("Starting the translator", "Test translation")
_REPLY_MAX = 40   # characters of the reply shown in the row
_ERROR_MAX = 160  # characters of a translate error shown in the alert


class _LlamaProbe(Protocol):
    def check_health(self) -> bool: ...
    def translate(self, text: str) -> str: ...


# Shipped default for translation.endpoint (default_config.yaml). Used only
# as the fallback when the merged-config read fails -- the production probe
# endpoint is resolved via _resolve_config_endpoint at page entry.
_DEFAULT_ENDPOINT = "http://localhost:8090/v1/chat/completions"


def _resolve_config_endpoint(default: str = _DEFAULT_ENDPOINT) -> str:
    """``translation.endpoint`` from the merged config (shipped defaults +
    user_overrides.yaml), falling back to the shipped default.

    The smoke probe must target the endpoint the RUNTIME will actually
    call: the watchdog honors translation.endpoint overrides (external
    server -> the bundled llama-server is not even spawned;
    loopback-with-port -> it binds there), and installer/README.md
    documents seeding user_overrides.yaml BEFORE first run. A probe
    hardcoded to the shipped default therefore showed a guaranteed false
    "health check failed" to every endpoint-override user forced through
    the wizard (v0.1.0 upgrades re-enter it after a model-pin change).

    A config-read failure falls back to ``default`` -- a broken
    user_overrides.yaml must degrade to today's behavior, never crash
    the wizard.
    """
    try:
        from yamabiko.config import load_full_config
        endpoint = (load_full_config().get("translation") or {}).get("endpoint")
    except Exception:
        _log.warning(
            "config read for translation.endpoint failed; probing the "
            "default %s", default, exc_info=True,
        )
        return default
    if isinstance(endpoint, str) and endpoint.strip():
        return endpoint.strip()
    return default


def _resolve_config_model(default: str = TRANSLATOR_ALIAS) -> str:
    """``translation.model`` from the merged config, falling back to the
    bundled alias.

    The runtime translator addresses every chat completion to the merged
    ``translation.model`` (translation/translator.py builds its request
    body from that key), so the smoke probe must send the same string: an
    external override endpoint (vLLM, Ollama, a proxied deployment) 404s
    the bundled llama-server alias, turning a working setup into a false
    "Translation test failed". Same degrade rule as
    ``_resolve_config_endpoint``: a config-read failure returns
    ``default`` rather than crashing the wizard.
    """
    try:
        from yamabiko.config import load_full_config
        model = (load_full_config().get("translation") or {}).get("model")
    except Exception:
        _log.warning(
            "config read for translation.model failed; probing with the "
            "default %s", default, exc_info=True,
        )
        return default
    if isinstance(model, str) and model.strip():
        return model.strip()
    return default


class TranslatorModelMismatch(RuntimeError):
    """The endpoint answered, but with a model other than the configured one.

    Another local program (LM Studio, Ollama, a second llama-server) already
    holds the port, so a reply from it says nothing about Yamabiko's own
    translator and must not pass the check. The message is the alert's
    first line.
    """

    def __init__(self, endpoint: str, model: str) -> None:
        where = urllib.parse.urlsplit(endpoint).netloc or endpoint
        super().__init__(f'Another app is answering on {where} (model "{model}").')
        self.endpoint = endpoint
        self.model = model


class TranslatorReplyError(RuntimeError):
    """The translator answered, but not with a translation: an empty answer
    or one that isn't the chat-completion JSON. The message is the alert's
    plain help line; the reply itself (``reply``) and what was wrong with it
    (the parse error, as ``__cause__``) go to the log."""

    def __init__(self, message: str, reply: object) -> None:
        super().__init__(message)
        self.reply = reply


class _HTTPLlamaProbe:
    """Production default — httpx probe.

    Spawning the bundled llama-server is the watchdog's job; this probe
    just verifies it's reachable and translating.
    """

    def __init__(
        self,
        endpoint: str = _DEFAULT_ENDPOINT,
        *,
        model: str = TRANSLATOR_ALIAS,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        parsed = urllib.parse.urlsplit(endpoint)
        # Health endpoint is at /health on the same host:port, regardless of
        # where the chat endpoint lives (handles /api prefixes, root deploys).
        self.health_url = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, "/health", "", "")
        )
        # The strict /health == 200 gate is only meaningful on the bundled
        # default endpoint: the watchdog owns that llama-server, which
        # serves 503 while loading and 200 once ready. External endpoints
        # may gate, redirect, or not implement /health at all while
        # serving chat completions fine -- see check_health.
        self._is_default_endpoint = endpoint == _DEFAULT_ENDPOINT
        # Whether the last /health attempt got any HTTP answer (Decision 131).
        self.answered = False

    def check_health(self) -> bool:
        # Any HTTP answer counts as "the server has started", a 503 while
        # llama-server loads its model included: the Translator step checks
        # "Starting the translator" once the endpoint answers at all
        # (Decision 131). Only a refused or timed-out request leaves it unset.
        self.answered = False
        try:
            r = loopback_http.get(self.health_url, timeout=5.0)
        except Exception:
            return False
        self.answered = True
        if r.status_code == 200:
            return True
        if self._is_default_endpoint:
            return False
        # External endpoint: mirror the runtime translator's readiness
        # policy (translation/translator.py wait_until_ready). 404/405 (no
        # /health route -- LM Studio-class servers, path-prefixed proxies)
        # and any other non-5xx (auth-gated / redirected /health) are NOT
        # loading signals: report alive and let the translate attempt
        # arbitrate. 5xx stays "still loading" (llama's own 503, or a
        # gateway 502/504 in front of a loading backend) and keeps the
        # poll going.
        return not 500 <= r.status_code < 600

    def translate(self, text: str) -> str:
        r = loopback_http.post(self.endpoint, json={
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Translate Japanese to English."},
                {"role": "user", "content": text},
            ],
            "max_tokens": 40, "temperature": 0.1, "stream": False,
            # Keep the wizard probe identical to the runtime translator's
            # Qwen request mode. Otherwise Qwen3.5 can spend this entire
            # small token budget in reasoning_content and return content="".
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=30.0)
        r.raise_for_status()
        try:
            data = r.json()
            replied_model = data.get("model")
        except (ValueError, AttributeError) as exc:  # not JSON, or not an object
            raise TranslatorReplyError(UNREADABLE_REPLY, r.text) from exc
        # An OpenAI-compatible server names the model that produced the
        # reply. llama-server answers with the alias Yamabiko configured;
        # any other name means a different service owns the endpoint.
        if replied_model is not None and replied_model != self.model:
            raise TranslatorModelMismatch(self.endpoint, str(replied_model))
        try:
            # A null content (no text at all) reads as an empty answer.
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise TranslatorReplyError(UNREADABLE_REPLY, data) from exc
        if not text:
            raise TranslatorReplyError(EMPTY_REPLY, data)
        return text


class _ProbeWorker(QObject):
    """Executes the blocking probe calls on the worker thread.

    Results carry the request's generation counter back so the page can
    discard responses that a Retry has superseded.
    """

    health_result = Signal(int, bool, bool)      # (generation, ok, answered)
    translate_result = Signal(int, bool, str)    # (generation, ok, text-or-error)
    # (generation, user-facing message): a different service answered.
    mismatch_result = Signal(int, str)

    def __init__(self, probe: _LlamaProbe) -> None:
        super().__init__()
        self._probe = probe

    @Slot(int)
    def do_health(self, gen: int) -> None:
        try:
            ok = bool(self._probe.check_health())
            # Any HTTP answer, a 503 while the model loads included (Decision
            # 131). A probe without ``answered`` (the tests' doubles)
            # answered exactly when it was healthy.
            answered = ok or bool(getattr(self._probe, "answered", False))
        except Exception:
            ok = answered = False
        self.health_result.emit(gen, ok, answered)

    @Slot(int)
    def do_translate(self, gen: int) -> None:
        try:
            text = self._probe.translate(SAMPLE_JP)
            if not isinstance(text, str) or not text.strip():
                raise TranslatorReplyError(EMPTY_REPLY, text)
        except TranslatorModelMismatch as exc:
            self.mismatch_result.emit(gen, str(exc))
            return
        except TranslatorReplyError as exc:
            # The alert gets the plain line; the log keeps the reply (cut
            # short) and what was wrong with it (the parse error, if any).
            _log.warning(
                "translator check: the reply is not a translation: reply=%.200r",
                exc.reply, exc_info=True,
            )
            self.translate_result.emit(gen, False, str(exc))
            return
        except Exception as exc:
            self.translate_result.emit(gen, False, str(exc))
            return
        self.translate_result.emit(gen, True, text)


def _sample_jp_font() -> QFont:
    font = QFont()
    font.setFamilies(["Yu Gothic UI", "Meiryo"])
    font.setPixelSize(15)
    return font


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def _error_line(payload: str) -> str:
    """A translate error for the alert: its first line, the user name
    removed, clipped (the log keeps the whole text)."""
    lines = redact_user(str(payload)).strip().splitlines()
    return _clip(lines[0] if lines else "", _ERROR_MAX)


class SmokePage(QWizardPage):
    """The Translator step. Retry and Skip exist only in the failure alert
    (L7); Skip emits ``skipped``, which the shell connects to ``next``."""

    # Requests to the worker thread (queued connections).
    _request_health = Signal(int)
    _request_translate = Signal(int)
    # The user chose Skip: the wizard goes on to Done.
    skipped = Signal()

    def __init__(
        self,
        *,
        probe: Optional[_LlamaProbe] = None,
        max_health_attempts: int = 15,
        health_poll_interval_ms: int = 2000,
        palette: dict = MIST,
    ) -> None:
        super().__init__()
        self.setTitle("Checking the translator")
        self.setSubTitle("A quick test to make sure it works.")
        # None means "no probe yet": the production probe is built lazily at
        # initializePage time so its endpoint reflects the MERGED config
        # (translation.endpoint override) rather than a construction-time
        # hardcode -- YamabikoWizard constructs this page bare, long before
        # the page is entered. Tests keep the injectable constructor seam
        # (and may also assign ``_probe`` directly before page entry; a
        # non-None probe is never replaced).
        self._probe: Optional[_LlamaProbe] = probe
        self._smoke_ok = False
        self._skipped = False
        # On first run the watchdog spawns llama-server in parallel and it
        # needs a multi-second CUDA warmup + GGUF load before binding :8090.
        # A single immediate health check therefore reliably hit
        # connection-refused and showed a false failure. Poll for up to
        # max_health_attempts * interval before declaring failure.
        self._max_health_attempts = max(1, int(max_health_attempts))
        self._health_poll_interval_ms = int(health_poll_interval_ms)
        self._health_attempts = 0
        # Whether the endpoint answered HTTP in this run (Decision 131).
        self._answered = False
        # Incremented on every run_smoke (and on cleanupPage and shutdown);
        # results tagged with an older generation are stale and ignored.
        self._gen = 0
        self._thread: Optional[QThread] = None
        self._worker: Optional[_ProbeWorker] = None
        self._port = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # The alert shows only on failure, 20 px under the lead; the rows sit
        # 22 px under whatever is above them (mockup).
        self.alert_box = QWidget()
        box = QVBoxLayout(self.alert_box)
        box.setContentsMargins(0, 20, 0, 0)
        self.alert = Alert(palette)
        box.addWidget(self.alert)
        self.retry_button = self.alert.add_button("Retry", primary=True)
        self.retry_button.clicked.connect(self.run_smoke)
        self.skip_button = self.alert.add_button("Skip")
        self.skip_button.clicked.connect(self._on_skip)
        self.alert_box.hide()
        layout.addWidget(self.alert_box)
        layout.addSpacing(22)
        self.rows = StatusRows(_ROW_NAMES, palette, row_height=52, sizes=False)
        self.rows.set_row(0, "waiting")
        self.rows.set_row(1, "waiting")
        layout.addWidget(self.rows)
        layout.addStretch(1)

    def initializePage(self) -> None:  # noqa: N802
        # Build the production probe here (not in __init__) so the endpoint
        # AND model honor translation.* overrides in the merged config --
        # see _resolve_config_endpoint / _resolve_config_model. An
        # injected/assigned probe is kept as-is.
        if self._probe is None:
            self._probe = _HTTPLlamaProbe(
                endpoint=_resolve_config_endpoint(),
                model=_resolve_config_model(),
            )
        self.run_smoke()

    def cleanupPage(self) -> None:  # noqa: N802 -- Qt convention
        # Back or a rail jump left the step: a new generation makes the
        # poll's pending timer and any result still in flight stale, so the
        # /health checks stop at once. The worker thread stays for a return,
        # where initializePage starts a fresh check.
        self._gen += 1
        super().cleanupPage()

    def _ensure_worker(self) -> None:
        """Lazily start the probe thread (idle event loop between calls)."""
        if self._thread is not None:
            return
        if self._probe is None:
            # Defensive: run_smoke called without initializePage (nothing in
            # production does this; kept so a stray Retry can't crash).
            self._probe = _HTTPLlamaProbe(
                endpoint=_resolve_config_endpoint(),
                model=_resolve_config_model(),
            )
        self._thread = QThread(self)
        self._worker = _ProbeWorker(self._probe)
        self._worker.moveToThread(self._thread)
        self._request_health.connect(self._worker.do_health)
        self._request_translate.connect(self._worker.do_translate)
        self._worker.health_result.connect(self._on_health_result)
        self._worker.translate_result.connect(self._on_translate_result)
        self._worker.mismatch_result.connect(self._on_model_mismatch)
        self._thread.start()

    def shutdown_thread(self, wait_ms: int = 2000) -> None:
        """Wizard-teardown hook: join the idle probe thread, or
        detach it into thread_keeper if a blocking httpx call is in
        flight. A detached probe thread exits on its own once the call's
        timeout (<= 30 s) fires; destroying it while running would be a
        Qt 6 qFatal abort.
        """
        thread, _worker = self._thread, self._worker
        self._thread = None
        self._worker = None
        self._gen += 1  # any in-flight result is now stale
        if thread is None:
            return
        thread.quit()
        if thread.wait(wait_ms):
            return
        thread_keeper.release(thread, _worker)

    def _endpoint_port(self) -> str:
        """host:port of the endpoint being checked ("localhost:8090")."""
        endpoint = getattr(self._probe, "endpoint", None) or _resolve_config_endpoint()
        return urllib.parse.urlsplit(endpoint).netloc or endpoint

    def _port_runs(self) -> tuple:
        return (Run(self._port, mono_font(13)),) if self._port else ()

    def run_smoke(self) -> None:
        # Reset BOTH states -- the user may retry after skipping.
        self._smoke_ok = False
        self._skipped = False
        self._health_attempts = 0
        self._answered = False
        self._gen += 1
        self._port = self._endpoint_port()
        self.alert_box.hide()
        self.rows.set_row(0, "now", self._port_runs())
        self.rows.set_row(1, "waiting")
        self.completeChanged.emit()
        self._ensure_worker()
        self._request_next_health_attempt()

    def _request_next_health_attempt(self) -> None:
        self._health_attempts += 1
        self._request_health.emit(self._gen)

    def _on_retry_timer(self, gen: int) -> None:
        if gen != self._gen:
            return  # superseded by a Retry/shutdown while the timer ran
        self._request_next_health_attempt()

    def _on_health_result(self, gen: int, ok: bool, answered: bool) -> None:
        if gen != self._gen:
            return  # stale result from a superseded attempt
        if ok:
            self.rows.set_row(0, "done", self._port_runs())
            self.rows.set_row(1, "now")
            self._request_translate.emit(self._gen)
            return
        if answered and not self._answered:
            # The server answers (llama-server serves 503 while it loads the
            # model): it has started, and the test translation waits for the
            # model (Decision 131).
            self._answered = True
            self.rows.set_row(0, "done", self._port_runs())
            self.rows.set_row(1, "now")
        if self._health_attempts >= self._max_health_attempts:
            if self._answered:
                self.rows.set_row(1, "paused")  # the mockup's picture
            else:
                self.rows.set_row(0, "paused", self._port_runs())  # nothing answered
            self._show_failure(NO_ANSWER, NO_ANSWER_HELP)
            return
        # Still starting -- the common first-run case, NOT a failure: the
        # rows keep their running marks while the poll goes on.
        gen_now = self._gen
        QTimer.singleShot(
            self._health_poll_interval_ms, lambda: self._on_retry_timer(gen_now),
        )

    def _on_translate_result(self, gen: int, ok: bool, payload: str) -> None:
        if gen != self._gen:
            return  # stale result from a superseded attempt
        if not ok:
            _log.warning("translator check: the test translation failed: %s", payload)
            self.rows.set_row(1, "paused")
            self._show_failure(TEST_FAILED, _error_line(payload))
            return
        font = ui_font(14)
        self.rows.set_row(1, "done", (
            Run(SAMPLE_JP, _sample_jp_font()),
            Run("\u2192", font),
            Run(_clip(payload, _REPLY_MAX), font),
        ))
        self._smoke_ok = True
        self.completeChanged.emit()

    def _on_model_mismatch(self, gen: int, message: str) -> None:
        if gen != self._gen:
            return  # stale result from a superseded attempt
        self.rows.set_row(1, "paused")
        self._show_failure(message, MISMATCH_HELP)

    def _show_failure(self, message: str, help_text: str) -> None:
        """The failure alert (L7): what went wrong, Retry, Skip and the skip
        note; Next stays disabled until a Retry passes or Skip."""
        self.alert.set_text(message, help_text, SKIP_NOTE)
        self.alert_box.show()
        self.completeChanged.emit()

    def _on_skip(self) -> None:
        self._skipped = True
        self.completeChanged.emit()
        self.skipped.emit()

    def isComplete(self) -> bool:  # noqa: N802
        return self._smoke_ok or self._skipped

    def smoke_ok(self) -> bool:
        return self._smoke_ok
