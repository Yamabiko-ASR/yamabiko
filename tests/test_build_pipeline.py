"""Wiring tests for yamabiko.main.build_pipeline().

yamabiko.main is split into:

    parse_args(argv) -> argparse.Namespace
    async build_pipeline(args) -> PipelineApp
    async run_pipeline(app) -> int
    async amain(args) -> None            (thin build+run composition)
    main(argv=None) -> None               (sync shell: argparse -> log
                                           setup -> wizard gate ->
                                           asyncio.run(amain(args)))

Tests here exercise build_pipeline with the heavy collaborators
(ASR / Translator / LoopbackCapture) stubbed. The fixture writes a
minimal user_overrides.yaml that disables romanization (glossary
file I/O) and hotkeys (Windows `keyboard` library seam) so
build_pipeline focuses on the core pipeline wiring.

Seeded defects (each listed assertion goes RED if the paired
production change is reverted):

- assert app.translator is not None:
    remove the Translator() construction in build_pipeline.
- assert app.asr is not None:
    remove the build_asr() call.
- assert app.chunker is not None:
    remove the SileroChunker construction.
- assert app.capture is not None:
    remove the LoopbackCapture construction.
- assert app.app_state.genre == "anime":
    drop the `genre=cfg.get("genre", "variety")` kwarg from
    AppState(...) in build_pipeline.
"""
from __future__ import annotations

import asyncio

import pytest

# Shared teardown-grace override for the two teardown-budget tests
# (test_run_pipeline_teardown_budget_bounds_*). It is written into the
# user_overrides.yaml lifecycle.grace_s.pipeline AND used as the
# wall-clock assert bound (GRACE_S + scheduler slack). Sourcing both
# from this one constant stops the override and the bound from drifting
# apart: bump the override here and the assert bound moves with it, so a
# stale bound can't mask a teardown that now overruns the new grace.
GRACE_S = 5.0
# Scheduler slack added on top of GRACE_S for the wall-clock upper bound.
_GRACE_ASSERT_SLACK_S = 3.0


class _FakeASR:
    def warmup(self):
        pass

    def transcribe(self, audio, sr):
        raise RuntimeError("test should not run ASR")


class _FakeTranslator:
    def __init__(self, *a, **kw):
        self.cfg = a[0] if a else None

    @property
    def health_check_url(self) -> str:
        # Used by HealthProbes refresher task, which runs in EVERY
        # run_pipeline-driving test because the WS server starts
        # unconditionally. The URL is a clearly-local sentinel; the
        # _pipeline_env fixture swaps httpx.AsyncClient for _FakeHttpClient
        # so the refresher NEVER opens a real socket on it (no live sockets
        # in the unit tier). The task is cancelled in run_pipeline's finally anyway.
        return "http://127.0.0.1:0/v1/models"

    readiness_url = "http://127.0.0.1:0/health"

    async def wait_until_ready(self, *, timeout_s, poll_s):
        return None

    async def sanity_check(self, genre):
        return None

    async def translate(self, *a, **kw):
        return ""

    async def aclose(self):
        return None


class _FakeHttpResponse:
    """Minimal stand-in for httpx.Response. _refresh_translator_health
    reads only ``.status_code``. 503 keeps ``_translator_reachable_cached``
    False (the honest unit-tier truth: there is no llama-server), without
    asserting on the value — these tests don't verify health, they just
    must not flake on a real socket.
    """
    status_code = 503


class _FakeHttpClient:
    """Stand-in for httpx.AsyncClient used by run_pipeline's
    ``_refresh_translator_health`` background task.

    No live sockets in the unit tier: with the real
    client, the refresher fired a real GET at the port-0 sentinel
    URL — a genuine connect() syscall (ConnectionRefused, swallowed by
    the production ``except Exception``) on EVERY run_pipeline-driving
    unit test, which previously triggered an httpx import-storm flake.
    This fake resolves ``.get`` without touching the network.

    It deliberately does NOT raise: the body of
    ``_refresh_translator_health`` runs to completion through this fake,
    so removing the production ``except Exception`` swallow would not
    crash any unit test — proving no unit test depends on a live GET
    (seeded-defect check).
    """

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, *a, **kw):
        return _FakeHttpResponse()


class _FakeCapture:
    def __init__(self, *a, pause_state_probe=None, **kw):
        self.started = False
        self.pause_edges = []
        # Mirror LoopbackCapture's production default: the idle-heartbeat pause
        # probe reads the constructor's pause_state_probe, and never pauses
        # without one. The pause-probe test's seeded defect (dropping that
        # build_pipeline wiring) then surfaces as the named 'paused stays
        # False' assertion, not an AttributeError.
        self._is_paused = (
            (lambda: pause_state_probe().paused)
            if pause_state_probe is not None
            else (lambda: False)
        )

    def apply_pause_edge(self, edge):
        self.pause_edges.append(edge)

    def start(self):
        self.started = True

    def stop(self):
        pass

    def endpoint_name(self):
        return None


class _FakeSileroVAD:
    """Stand-in for vad.silero_chunker.SileroVAD.

    Lets the REAL SileroChunker be constructed in the unit tier without
    (a) resolving the model path via ``_default_model_path()`` ->
    ``hf_hub_download(onnx-community/silero-vad)`` -- a live HuggingFace
    Hub call -- or (b) loading an onnxruntime session. See the
    _pipeline_env monkeypatch for the why. The real download + signature
    path stays covered by the @pytest.mark.real tier
    (tests/test_real_hf_signature.py).
    """

    def __init__(self, model_path, sample_rate: int = 16000) -> None:
        self.model_path = model_path
        self.sample_rate = sample_rate

    def __call__(self, frame) -> float:
        return 0.0

    def reset_states(self) -> None:
        pass


class _FakeSoundTagger:
    """Stand-in for vad.silero_chunker.SoundTagger: hears nothing."""

    def __init__(self, model_path) -> None:
        self.model_path = model_path

    def __call__(self, window):
        import numpy as np

        return np.zeros(250, dtype=np.float32)


class _NoOpenWindow:
    """What run_pipeline's ASR stage reads from a stand-in chunker
    (asr.incremental.OpenWindowSource): no window is ever open."""

    decided_samples = 0
    open_window_id = None

    def open_window_snapshot(self):
        return None


def _write_overrides(tmp_path, extra: dict | None = None) -> None:
    """Write a user_overrides.yaml that disables the heavy features
    so build_pipeline focuses on core wiring. Extra keys merge in.
    """
    import yaml
    overrides_dir = tmp_path / "Yamabiko"
    overrides_dir.mkdir(parents=True, exist_ok=True)
    data: dict = {
        "romanization": {"enabled": False},
        "hotkeys": {"enabled": False},
        "overlay": {"browser": False},
    }
    if extra:
        # Shallow-merge under each top-level key; extras win.
        for k, v in extra.items():
            if isinstance(v, dict) and isinstance(data.get(k), dict):
                data[k] = {**data[k], **v}
            else:
                data[k] = v
    (overrides_dir / "user_overrides.yaml").write_text(
        yaml.safe_dump(data), encoding="utf-8",
    )


class _FakeGlossaryManager:
    """Stand-in for GlossaryManager so build_pipeline's romanization
    branch doesn't hit YAML I/O or require bundled glossary files.
    """
    def __init__(self, *a, **kw):
        pass

    def current_genre(self):
        return "variety"

    def set_genre(self, genre):
        pass

    def save(self):
        pass


class _FakeRomanizationPipeline:
    def __init__(self, *a, **kw):
        # Capture kwargs so ctor-wiring tests can introspect what
        # build_pipeline passed. The ctor no longer takes a gazetteer --
        # the trio (gazetteer, media_id, epoch) arrives via set_gazetteer,
        # the single write path, tracked below.
        self.init_kwargs = kw
        # Mirror the real RomanizationPipeline's single-write-path contract:
        # the gazetteer, its media_id, and the applied epoch all start empty
        # and are stamped together by set_gazetteer, so boot-seed / hot-loop
        # tests observe exactly what was applied.
        self._gazetteer = None
        self._series_id = None
        self._applied_epoch = 0

    def set_gazetteer(self, gazetteer, media_id, epoch=0):
        self._gazetteer = gazetteer
        self._series_id = media_id
        self._applied_epoch = epoch

    def current_series_id(self):
        return self._series_id

    def applied_series_epoch(self):
        return self._applied_epoch


class _FakeHotkeyManager:
    """Stand-in for yamabiko.hotkeys.HotkeyManager. Records the
    kwargs passed to register() so hotkey-wiring tests can introspect
    which keybinds and callbacks made it through build_pipeline ->
    run_pipeline, and every register/unregister call in order so a
    re-bind test can tell "cleared, then bound" from the reverse.
    """
    def __init__(self):
        self.register_calls: list[dict] = []
        self.calls: list[str] = []

    def register(self, **kwargs):
        self.calls.append("register")
        self.register_calls.append(kwargs)

    def unregister(self):
        self.calls.append("unregister")


class _FakeFastAPIManager:
    """Stand-in for display.broadcast.ConnectionManager on the
    FastAPI state. Tests assert identity (`app.manager is
    env.fake_fastapi_app.state.manager`) to prove build_app's
    return was wired through. broadcast() is an awaitable that
    records each call so hotkey-behaviour tests can inspect the
    payloads.
    """
    def __init__(self):
        self.broadcasts: list[dict] = []

    clients = 0
    native_clients = 0

    def client_count(self) -> int:
        return self.clients

    def native_client_count(self) -> int:
        return self.native_clients

    async def broadcast(self, payload: dict) -> None:
        # Intentionally a deep-copy-less record: tests assert on
        # identity and on individual keys, not on long-term
        # immutability.
        self.broadcasts.append(payload)


class _FakeFastAPIState:
    def __init__(self):
        self.manager = _FakeFastAPIManager()


class _FakeFastAPIApp:
    """Stand-in for the FastAPI app that build_app returns in the
    real code. Only needs a .state with a .manager attribute; that's
    all run_pipeline reads.
    """
    def __init__(self):
        self.state = _FakeFastAPIState()


class _FakeUvicornConfig:
    def __init__(self, app, host, port, log_level, access_log, **kwargs):
        # Accept arbitrary extra kwargs (e.g. timeout_graceful_shutdown,
        # lifespan) so production Config() calls that add new tuning
        # knobs don't break these stage-wiring tests.
        self.app = app
        self.host = host
        self.port = port
        self.log_level = log_level
        self.access_log = access_log
        # Mirror Uvicorn 0.44.0's real defaults so deleting an explicit
        # production selector makes the wiring test fail with ``"auto"``.
        self.loop = kwargs.get("loop", "auto")
        self.http = kwargs.get("http", "auto")
        self.ws = kwargs.get("ws", "auto")
        self.lifespan = kwargs.get("lifespan", "auto")
        self.extra_kwargs = kwargs


class _FakeUvicornServer:
    """Stand-in for uvicorn.Server. serve() no-ops until
    should_exit, matching the interface run_pipeline relies on.
    """
    def __init__(self, config):
        self.config = config
        # install_signal_handlers starts as a real callable (default
        # behaviour); run_pipeline overwrites it with `lambda: None`
        # to stop uvicorn from stealing SIGINT. Tests assert the
        # overwrite happened by checking it's callable (the default
        # None sentinel below would flip to a real callable).
        self.install_signal_handlers = "default-method-placeholder"
        self.should_exit = False

    async def serve(self):
        # Loop until should_exit OR cancelled. In the tests,
        # run_pipeline sets should_exit during its finally clause;
        # we cooperate so the asyncio.wait_for(..., timeout=3.0)
        # path exits cleanly without the CancelledError branch.
        while not self.should_exit:
            await asyncio.sleep(0.01)


class _PipelineEnv:
    """Return value of the _pipeline_env fixture.

    Holds the module under test AND references to each faked
    collaborator. Tests use the stored references to assert IDENTITY
    (`app.asr is env.fake_asr`) rather than mere `is not None`.
    Without this, a Liar-pattern weakness creeps in: `is not None`
    passes even if someone hardcoded a fresh fake inside build_pipeline
    without going through the monkeypatched factory seam.
    """
    def __init__(self, module, fake_asr, translator_instances,
                 capture_instances,
                 glossary_instances, romanization_instances,
                 fake_hotkey_mgr, fake_fastapi_app,
                 build_app_calls, uvicorn_server_instances):
        self.module = module
        self.fake_asr = fake_asr
        self.translator_instances = translator_instances
        self.capture_instances = capture_instances
        self.glossary_instances = glossary_instances
        self.romanization_instances = romanization_instances
        self.fake_hotkey_mgr = fake_hotkey_mgr
        self.fake_fastapi_app = fake_fastapi_app
        self.build_app_calls = build_app_calls
        self.uvicorn_server_instances = uvicorn_server_instances


@pytest.fixture
def _pipeline_env(monkeypatch, tmp_path):
    """Minimal env for build_pipeline(): inject fakes for the heavy
    collaborators (ASR / Translator / Capture) so the call doesn't
    allocate GPU / open audio device / hit llama.

    Returns a _PipelineEnv holding references to each fake so tests
    can assert identity against the objects build_pipeline actually
    wired in.
    """
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    import yamabiko.main as _main_mod

    # build_asr is a plain factory function; stub to return a known
    # _FakeASR instance so the test can assert app.asr is fake_asr.
    fake_asr = _FakeASR()
    monkeypatch.setattr(_main_mod, "build_asr", lambda cfg: fake_asr)

    # Translator and LoopbackCapture are classes instantiated inside
    # build_pipeline. Track the instances the refactored code
    # constructs so tests can assert identity against them.
    translator_instances: list = []
    capture_instances: list = []

    class _TrackedTranslator(_FakeTranslator):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            translator_instances.append(self)

    class _TrackedCapture(_FakeCapture):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            capture_instances.append(self)

    monkeypatch.setattr(_main_mod, "Translator", _TrackedTranslator)
    monkeypatch.setattr(_main_mod, "LoopbackCapture", _TrackedCapture)

    # SileroChunker is constructed directly inside build_pipeline with
    # no injectable seam (see test_build_pipeline_wires_core_collaborators).
    # Its __init__ resolves the model path via _default_model_path() ->
    # hf_hub_download(onnx-community/silero-vad) -- a LIVE HuggingFace Hub
    # call -- then loads an onnxruntime session. In the unit tier
    # (-m "not real") that is a network dependency: a cold CI cache + an
    # HF rate-limit returns 429. Isolate BOTH boundaries so build_pipeline
    # still constructs the REAL SileroChunker class -- keeping
    # isinstance(app.chunker, SileroChunker) and its field-swap
    # seeded-defect meaningful -- while neither hitting the network nor
    # loading onnx. The real HF path stays covered by the
    # @pytest.mark.real tier (tests/test_real_hf_signature.py).
    import yamabiko.vad.silero_chunker as _silero_mod
    monkeypatch.setattr(
        _silero_mod, "_default_model_path",
        lambda: tmp_path / "fake-silero-vad.onnx",
    )
    monkeypatch.setattr(_silero_mod, "SileroVAD", _FakeSileroVAD)
    # Same reason for the sound tagger of the VAD assist: no onnxruntime
    # session in the unit tier (tests/test_real_sound_tagger.py covers it).
    monkeypatch.setattr(_silero_mod, "SoundTagger", _FakeSoundTagger)

    # GlossaryManager + RomanizationPipeline: stub construction so
    # the romanization branch in build_pipeline wires without
    # needing bundled glossary YAML files to exist in the tmp env.
    glossary_instances: list = []
    romanization_instances: list = []

    class _TrackedGlossaryManager(_FakeGlossaryManager):
        def __init__(self, gdir, *a, **kw):
            super().__init__(gdir, *a, **kw)
            # Store the gdir + seed_path build_pipeline resolved to, so
            # path-fallback / seed-layer tests can verify resolution.
            self.gdir = gdir
            self.seed_path = kw.get("seed_path")
            glossary_instances.append(self)

    class _TrackedRomanizationPipeline(_FakeRomanizationPipeline):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            romanization_instances.append(self)

    monkeypatch.setattr(_main_mod, "GlossaryManager", _TrackedGlossaryManager)
    monkeypatch.setattr(
        _main_mod, "RomanizationPipeline", _TrackedRomanizationPipeline,
    )
    # Romanizer has no mutable state in construction; no need to
    # track instances, just swap for a cheap stand-in.
    monkeypatch.setattr(_main_mod, "Romanizer", lambda: object())

    # HotkeyManager: stub on the hotkeys module so run_pipeline's
    # `hk_mod.HotkeyManager()` returns the fixture-owned fake. The
    # test drives run_pipeline (not just build_pipeline) to reach
    # the hotkey registration path, which lives in run_pipeline next
    # to the browser server (PipelineApp.register_hotkeys binds the
    # _on_*_hotkey adapters, which read app.manager at fire time).
    import yamabiko.hotkeys as _hk_mod
    fake_hotkey_mgr = _FakeHotkeyManager()
    monkeypatch.setattr(_hk_mod, "HotkeyManager", lambda: fake_hotkey_mgr)

    # Browser server: stub build_app and uvicorn.Config/Server so
    # run_pipeline's browser branch wires without actually binding a
    # port or constructing FastAPI middleware. Records each call so
    # wiring tests can assert what was passed.
    fake_fastapi_app = _FakeFastAPIApp()
    build_app_calls: list[dict] = []

    def _tracking_build_app(state, on_quit, health_probes=None,
                            serve_browser_ui=True, *, on_pause_change=None,
                            on_hotkeys_change=None,
                            bound_port=None, ws_token=None,
                            gazetteer_enabled=True):
        build_app_calls.append({
            "state": state,
            "on_quit": on_quit,
            "on_pause_change": on_pause_change,
            "on_hotkeys_change": on_hotkeys_change,
            "health_probes": health_probes,
            "serve_browser_ui": serve_browser_ui,
            "bound_port": bound_port,
            "ws_token": ws_token,
            "gazetteer_enabled": gazetteer_enabled,
        })
        return fake_fastapi_app

    monkeypatch.setattr(_main_mod, "build_app", _tracking_build_app)

    uvicorn_server_instances: list = []

    class _TrackedUvicornServer(_FakeUvicornServer):
        def __init__(self, config):
            super().__init__(config)
            uvicorn_server_instances.append(self)

    import uvicorn
    monkeypatch.setattr(uvicorn, "Config", _FakeUvicornConfig)
    monkeypatch.setattr(uvicorn, "Server", _TrackedUvicornServer)

    # run_pipeline's _refresh_translator_health background task opens an
    # httpx.AsyncClient and GETs app.translator.health_check_url on a
    # 10 s loop. _FakeTranslator's port-0 sentinel URL would still drive
    # a REAL connect() syscall (ConnectionRefused, swallowed by the
    # production except Exception) on every run_pipeline-driving test —
    # a live socket in the unit tier + the source of a prior httpx
    # import-storm flake. Swap the client for a socket-free fake so the
    # refresher's body still runs end-to-end (the seeded-defect check
    # stays meaningful) but never touches the network. main.py builds it
    # via loopback_http.async_client, which reads `httpx.AsyncClient` at
    # call time, so patching the module attribute covers the call site.
    # httpx.AsyncHTTPTransport (used directly by the teardown-budget tests
    # to pre-pay the lazy import) is a different class and stays real.
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _FakeHttpClient)

    # Skip rotating file log + latency log (idempotent already but
    # noisy across tests).
    import yamabiko.log_config as _lc
    monkeypatch.setattr(_lc, "setup_latency_log", lambda enabled=True: None)
    monkeypatch.setattr(_lc, "setup_main_log", lambda *a, **kw: None)

    # Stub the overlay subprocess + inner Job Object for EVERY test in
    # this file (test isolation). Without this in the
    # fixture, run_pipeline-driving tests briefly Popen'd a REAL Qt
    # overlay child (python -m yamabiko.overlay_app) via the overlay
    # supervisor thread — racy, slow, and noisy (overlay.mainwindow.shown
    # + ws_client reconnect warnings in captured output), and able to
    # orphan GUI processes if pytest dies mid-teardown. Tests that need
    # different process fakes (e.g. the spawn-argv test) re-patch
    # over this via their own monkeypatch — last patch wins. The real
    # overlay-spawn seam stays covered at the bundle tier by
    # tests/test_real_frozen_e2e.py (default-launch process tree).
    _stub_overlay_runtime(monkeypatch)

    return _PipelineEnv(
        module=_main_mod,
        fake_asr=fake_asr,
        translator_instances=translator_instances,
        capture_instances=capture_instances,
        glossary_instances=glossary_instances,
        romanization_instances=romanization_instances,
        fake_hotkey_mgr=fake_hotkey_mgr,
        fake_fastapi_app=fake_fastapi_app,
        build_app_calls=build_app_calls,
        uvicorn_server_instances=uvicorn_server_instances,
    )


def test_build_pipeline_wires_core_collaborators(_pipeline_env, tmp_path):
    """build_pipeline(args) returns a PipelineApp wired to the
    collaborators produced by the injected factories.

    Identity assertions (`app.asr is env.fake_asr`, etc.) are stronger
    than `is not None`: they prove the field came THROUGH the factory
    seam we control, not from a hardcoded fallback inside build_pipeline.
    That retires a Liar-pattern weakness where a naive `is not None`
    would pass even if someone wrote `app.asr = _FakeASR()` inline.

    For chunker and app_state, which are constructed directly inside
    build_pipeline with no injectable seam, assert against the CONCRETE
    TYPE (isinstance) so a silent field-swap in the dataclass return
    still fails the test.

    Seeded-defect checks:
      - build_pipeline never calling build_asr / Translator() /
        LoopbackCapture() -> field is None or instances list is empty
        -> identity assertion fails.
      - build_pipeline swapping `chunker` and `app_state` positions
        in the dataclass return -> isinstance assertion fails with
        a wrong-type message.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)

    args = env.module.parse_args(["--max-segments", "0"])
    app = asyncio.run(env.module.build_pipeline(args))

    # Identity: the faked factories were called and their outputs
    # are what got wired.
    assert app.asr is env.fake_asr, (
        "app.asr is not the instance returned by build_asr; either "
        "build_asr wasn't called or the return value wasn't stored"
    )
    assert len(env.translator_instances) == 1, (
        f"expected exactly one Translator construction; got "
        f"{len(env.translator_instances)}"
    )
    assert app.translator is env.translator_instances[0], (
        "app.translator is not the Translator instance constructed "
        "inside build_pipeline"
    )
    assert len(env.capture_instances) == 1, (
        f"expected exactly one LoopbackCapture construction; got "
        f"{len(env.capture_instances)}"
    )
    assert app.capture is env.capture_instances[0], (
        "app.capture is not the LoopbackCapture instance constructed "
        "inside build_pipeline"
    )

    # Type: chunker and app_state have no injectable seam; isinstance
    # still catches a field-swap in the dataclass return.
    from yamabiko.vad.silero_chunker import SileroChunker
    from yamabiko.state import AppState
    assert isinstance(app.chunker, SileroChunker), (
        f"app.chunker is not a SileroChunker; got {type(app.chunker).__name__}"
    )
    assert isinstance(app.app_state, AppState), (
        f"app.app_state is not an AppState; got {type(app.app_state).__name__}"
    )
    assert isinstance(app.cfg, dict), (
        f"app.cfg is not a dict; got {type(app.cfg).__name__}"
    )


def test_build_pipeline_plumbs_vad_dip_split_config_into_gate(
    _pipeline_env, tmp_path,
):
    """build_pipeline must carry the vad.dip_split_* keys through
    ChunkerConfig into the live FrameGate, not silently drop them (which
    would leave the shipped feature off / mis-tuned in production).

    Asserts against the constructed gate's derived state, so it fails if
    the ChunkerConfig(...) call build_pipeline uses
    (vad.chunker.chunker_config_from_config) omits any dip kwarg.

    Seeded-defect check: delete `dip_split_enabled=vad.get(...)`
    from chunker_config_from_config -> the gate uses the dataclass
    default (False) -> `_dip_enabled is True` assert red. Delete the
    min_piece kwarg -> the frames assert (1400ms -> ~43 frames) red.
    """
    env = _pipeline_env
    _write_overrides(tmp_path, {"vad": {
        "dip_split_enabled": True,
        "dip_split_threshold": 0.22,
        "dip_split_min_piece_ms": 1400,
    }})

    args = env.module.parse_args(["--max-segments", "0"])
    app = asyncio.run(env.module.build_pipeline(args))

    gate = app.chunker._gate
    assert gate._dip_enabled is True, (
        "vad.dip_split_enabled did not reach the gate"
    )
    assert gate._dip_threshold == 0.22, (
        f"vad.dip_split_threshold not plumbed: {gate._dip_threshold}"
    )
    # 1400 ms / 32 ms per frame == 43 frames (int floor).
    assert gate._dip_min_piece_frames == 43, (
        f"vad.dip_split_min_piece_ms not plumbed: "
        f"{gate._dip_min_piece_frames} frames"
    )


def test_build_pipeline_dropped_key_fallbacks_match_shipped_defaults(
    _pipeline_env, tmp_path, monkeypatch,
):
    """P2 (2026-07-12 review): several .get() fallbacks in build_pipeline
    contradicted the shipped default_config.yaml values. Dead today (the
    layered load always restores every key), but if a key were ever
    dropped from the merged config, the wrong fallback would silently
    re-tune VAD (0.5 vs the shipped 0.3) or turn a shipped-ON feature OFF
    (romanization / gazetteer fuzzy recovery / polarity guard).

    The merged config is loaded for real, then the four keys are deleted
    via a wrapped _load_config, making the fallbacks live.

    Seeded defect: flip any one fallback back to its old contradicting
    value (e.g. `cfg["vad"].get("threshold", 0.5)`) -> the matching assert
    goes red.
    """
    env = _pipeline_env
    # Fixture default disables romanization via overrides; deleting the
    # merged key below hands the decision to the fallback under test.
    _write_overrides(tmp_path)

    real_load = env.module._load_config

    def _load_then_drop_keys(explicit):
        cfg = real_load(explicit)
        del cfg["vad"]["threshold"]
        del cfg["romanization"]["enabled"]
        del cfg["gazetteer"]["fuzzy_enabled"]
        del cfg["translation"]["polarity_guard"]
        return cfg

    monkeypatch.setattr(env.module, "_load_config", _load_then_drop_keys)

    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))

    # default_config.yaml vad.threshold: 0.3 (produced-media tuning), NOT
    # Silero's generic 0.5. The chunker forwards threshold into its
    # FrameGate (the same observable the dip-split plumbing test reads),
    # so assert on the gate the pipeline will actually run with.
    assert app.chunker._gate.threshold == 0.3, (
        f"vad.threshold fallback disagrees with the shipped 0.3 default; "
        f"got {app.chunker._gate.threshold}"
    )
    # default_config.yaml romanization.enabled: true.
    assert app.romanization_pipeline is not None, (
        "romanization.enabled fallback turned the shipped-ON feature OFF"
    )
    # default_config.yaml gazetteer.fuzzy_enabled: true.
    assert env.romanization_instances[0].init_kwargs["fuzzy_enabled"] is True, (
        "gazetteer.fuzzy_enabled fallback turned the shipped-ON feature OFF"
    )
    # default_config.yaml translation.polarity_guard: true.
    assert app.translator.cfg.polarity_guard is True, (
        "translation.polarity_guard fallback turned the shipped-ON guard OFF"
    )


def test_build_pipeline_applies_cfg_genre_to_app_state(_pipeline_env, tmp_path):
    """build_pipeline must seed AppState.genre from cfg['genre'].
    Without this wire, the first translated segment uses the default
    'variety' genre regardless of what the user configured.

    Seeded-defect check: drop the `genre=cfg.get("genre", "variety")`
    kwarg from AppState(...) -> AppState defaults to 'variety',
    this test fails with "got 'variety'" rather than 'anime'.
    """
    env = _pipeline_env
    _write_overrides(tmp_path, {"genre": "anime"})

    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))
    assert app.app_state.genre == "anime", (
        f"cfg['genre']='anime' did not reach AppState; "
        f"got {app.app_state.genre!r}"
    )


def test_build_pipeline_forces_16k_sample_rate_on_unsupported_override(
    _pipeline_env, tmp_path, caplog,
):
    """Yamabiko's Silero wrapper contract is fixed at 16 kHz with
    512-sample external frames. The wrapper prepends 64 context samples,
    so the raw ONNX graph receives 576 samples. An audio.sample_rate
    override of 48000 passes the config type validator (it's a valid
    int) and used to flow straight into AudioConfig + ChunkerConfig,
    violating the model contract at inference time. build_pipeline must
    coerce to 16000 and warn, naming the key.

    Seeded defect (one line): delete the sample-rate guard in
    build_pipeline -> app.audio_cfg.sample_rate == 48000 -> red.
    """
    import logging as _logging

    env = _pipeline_env
    _write_overrides(tmp_path, {"audio": {"sample_rate": 48000}})

    args = env.module.parse_args([])
    with caplog.at_level(_logging.WARNING, logger="yamabiko"):
        app = asyncio.run(env.module.build_pipeline(args))

    assert app.audio_cfg.sample_rate == 16000, (
        f"audio.sample_rate=48000 reached AudioConfig "
        f"(got {app.audio_cfg.sample_rate}); the Silero 512-sample "
        f"external-frame/16kHz contract is violated at inference time"
    )
    assert app.chunker._cfg.sample_rate == 16000, (
        f"audio.sample_rate=48000 reached ChunkerConfig "
        f"(got {app.chunker._cfg.sample_rate})"
    )
    warnings = [
        r.getMessage() for r in caplog.records
        if r.levelno >= _logging.WARNING and "sample_rate" in r.getMessage()
    ]
    assert warnings, (
        "no WARNING naming audio.sample_rate was logged; the override "
        "was coerced (or dropped) silently"
    )
    assert any("16000" in m for m in warnings), (
        f"the warning must state the forced value 16000; got {warnings!r}"
    )


def test_build_pipeline_wires_romanization_when_enabled(
    _pipeline_env, tmp_path,
):
    """When cfg['romanization']['enabled'] is True, build_pipeline
    must construct a GlossaryManager + RomanizationPipeline and
    store both on PipelineApp. When disabled (the fixture default),
    both must be None.

    The base fixture disables romanization so this branch was previously
    unexercised.

    Seeded-defect check: in build_pipeline, change
        romanization_pipeline = RomanizationPipeline(glossary_mgr, Romanizer())
    to
        romanization_pipeline = None  # forget
    -> identity assertion below fails because the list tracking
    constructed instances has length 1 but app.romanization_pipeline
    is None.
    """
    env = _pipeline_env
    # Point glossaries_dir at a temp dir so the resolver doesn't
    # depend on the bundled package path.
    gdir = tmp_path / "test-glossaries"
    gdir.mkdir()
    _write_overrides(tmp_path, {
        "romanization": {"enabled": True, "glossaries_dir": str(gdir)},
    })

    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))

    assert len(env.glossary_instances) == 1, (
        f"expected one GlossaryManager construction; got "
        f"{len(env.glossary_instances)}"
    )
    assert app.glossary_mgr is env.glossary_instances[0], (
        "app.glossary_mgr is not the GlossaryManager instance "
        "constructed inside build_pipeline"
    )
    assert len(env.romanization_instances) == 1, (
        f"expected one RomanizationPipeline construction; got "
        f"{len(env.romanization_instances)}"
    )
    assert app.romanization_pipeline is env.romanization_instances[0], (
        "app.romanization_pipeline is not the RomanizationPipeline "
        "instance constructed inside build_pipeline"
    )


def test_build_pipeline_reads_seed_layer_from_bundle_when_no_override(
    _pipeline_env, tmp_path,
):
    """The seed layer must be read from the read-only BUNDLE copy,
    not from the copy-if-absent app-data dir — otherwise a revised
    shipped _seed.yaml never reaches an existing install (the seed file
    in app-data is frozen at first-install version).

    Seeded defect: revert build_pipeline to `seed_path = gdir / "_seed.yaml"`
    on the no-override path -> gdir is the LOCALAPPDATA glossaries dir,
    not the bundle dir -> red.
    """
    from yamabiko import paths

    env = _pipeline_env
    _write_overrides(tmp_path, {"romanization": {"enabled": True}})

    args = env.module.parse_args([])
    asyncio.run(env.module.build_pipeline(args))

    assert len(env.glossary_instances) == 1
    mgr = env.glossary_instances[0]
    assert mgr.seed_path == paths.bundle_data_dir() / "glossaries" / "_seed.yaml", (
        f"seed layer must come from the bundle; got {mgr.seed_path}"
    )
    # The writable dir is still app-data — only the seed layer moved.
    assert mgr.gdir == paths.app_data_dir() / "glossaries"


def test_build_pipeline_override_dir_keeps_local_seed_yaml(_pipeline_env, tmp_path):
    """An explicit glossaries_dir override is fully user-managed: the
    seed layer stays inside the override dir (user knows best)."""
    env = _pipeline_env
    gdir = tmp_path / "my-glossaries"
    gdir.mkdir()
    _write_overrides(tmp_path, {
        "romanization": {"enabled": True, "glossaries_dir": str(gdir)},
    })

    args = env.module.parse_args([])
    asyncio.run(env.module.build_pipeline(args))

    assert env.glossary_instances[0].seed_path == gdir / "_seed.yaml"


def test_build_pipeline_romanization_none_when_disabled(_pipeline_env, tmp_path):
    """Paired negative case: romanization.enabled=False leaves
    glossary_mgr and romanization_pipeline as None.

    Seeded-defect check: remove the
        `if rom_cfg.get("enabled", False):`
    guard in build_pipeline -> glossary construction happens anyway,
    this assertion fails.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)  # fixture default disables romanization

    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))

    assert app.glossary_mgr is None, (
        f"glossary_mgr must be None when romanization disabled; "
        f"got {app.glossary_mgr!r}"
    )
    assert app.romanization_pipeline is None, (
        f"romanization_pipeline must be None when disabled; "
        f"got {app.romanization_pipeline!r}"
    )


def test_build_pipeline_restores_gazetteer_when_enabled_with_cached_series(
    _pipeline_env, tmp_path,
):
    """Boot-seed, cache PRESENT: gazetteer.enabled + a persisted
    active_series.json pointer + a saved per-series gazetteer cache ->
    (1) the pipeline is handed that SAME gazetteer via set_gazetteer
    (proven via a real lookup(), not just identity/None), (2) AppState is
    seeded with the pointer's media_id, and (3) the pipeline's series id is
    stamped so the hot loop doesn't redundantly reload on sentence one.

    Seeded defect: in build_pipeline, drop the inline
    `set_gazetteer(_restored_gaz, _sid, epoch=...)` call -> the
    `gaz is not None` and current_series_id() asserts go red. Drop the inline
    `app_state.set_series(...)` -> the series_media_id assert goes red.
    """
    import json

    from yamabiko import paths
    from yamabiko.gazetteer.gazetteer import SeriesGazetteer, gazetteer_path

    env = _pipeline_env
    _write_overrides(tmp_path, {
        "romanization": {"enabled": True},
        "gazetteer": {"enabled": True},
    })
    paths.ensure_dirs()
    (paths.app_data_dir() / "active_series.json").write_text(
        json.dumps({"media_id": 4224, "label": "Test Series"}),
        encoding="utf-8",
    )
    SeriesGazetteer({"レックス": "Rex"}).save(gazetteer_path(4224))

    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))

    assert len(env.romanization_instances) == 1, (
        f"expected one RomanizationPipeline construction; got "
        f"{len(env.romanization_instances)}"
    )
    gaz = env.romanization_instances[0]._gazetteer
    assert gaz is not None, (
        "gazetteer was not applied (via set_gazetteer) from active_series.json "
        "+ the per-series gazetteer cache"
    )
    assert gaz.lookup("レックス") == "Rex", (
        f"restored gazetteer did not round-trip the cached lookup; "
        f"got {gaz.lookup('レックス')!r}"
    )
    # (2) runtime state seeded from the pointer so a freshly-connected
    # overlay shows the restored series.
    assert app.app_state.series_media_id == 4224, (
        f"boot-seed did not stamp AppState.series_media_id from the pointer; "
        f"got {app.app_state.series_media_id!r}"
    )
    assert app.app_state.series_label == "Test Series"
    # (3) pipeline series id stamped -> hot loop won't reload sentence one.
    assert app.romanization_pipeline.current_series_id() == 4224, (
        "boot-seed did not stamp the pipeline's series id via set_gazetteer; "
        "the hot loop would reload the gazetteer on the first sentence"
    )


def test_build_pipeline_gazetteer_none_when_no_active_series_file(
    _pipeline_env, tmp_path,
):
    """Boot-seed, NO pointer file: gazetteer.enabled but NO
    active_series.json written yet (fresh install, user hasn't picked a
    series) -> the pipeline still gets constructed, but with gazetteer=None
    and no series seeded -- a missing pointer file is a soft no-op, not a
    startup failure.

    Seeded defect: make load_active_series raise instead of returning None
    on a missing pointer file -> build_pipeline's whole romanization
    try/except catches it and BOTH glossary_mgr and romanization_pipeline
    degrade to None -> the `len(env.romanization_instances) == 1` assertion
    goes red (proving the missing-file path must be handled inside the
    loader, not by the outer degrade-on-exception net).
    """
    from yamabiko import paths

    env = _pipeline_env
    _write_overrides(tmp_path, {
        "romanization": {"enabled": True},
        "gazetteer": {"enabled": True},
    })
    paths.ensure_dirs()
    pointer = paths.app_data_dir() / "active_series.json"
    assert not pointer.exists(), "test setup: pointer file must be absent"

    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))

    assert len(env.romanization_instances) == 1
    gaz = env.romanization_instances[0]._gazetteer
    assert gaz is None, (
        f"expected no gazetteer applied with no active_series.json; got {gaz!r}"
    )
    # No pointer -> no series seeded; the pipeline's series id stays None.
    assert app.app_state.series_media_id is None, (
        f"series must not be seeded with no pointer file; got "
        f"{app.app_state.series_media_id!r}"
    )
    assert app.romanization_pipeline.current_series_id() is None


def test_build_pipeline_seeds_series_but_gazetteer_none_when_cache_absent(
    _pipeline_env, tmp_path,
):
    """Boot-seed, pointer PRESENT but cache ABSENT: active_series.json names a
    series whose per-series gazetteer file was never saved -> the series is
    STILL seeded onto AppState and the pipeline's series id is STILL stamped
    (so the hot loop won't retry the load every sentence), but the gazetteer
    is None (better no hints than wrong-series names).

    Seeded defect: gate `app_state.set_series` / the boot `set_gazetteer` on a
    gazetteer having loaded (i.e. only seed when cache present) -> the
    series_media_id / current_series_id() asserts go red.
    """
    import json

    from yamabiko import paths

    env = _pipeline_env
    _write_overrides(tmp_path, {
        "romanization": {"enabled": True},
        "gazetteer": {"enabled": True},
    })
    paths.ensure_dirs()
    (paths.app_data_dir() / "active_series.json").write_text(
        json.dumps({"media_id": 4224, "label": "Test Series"}),
        encoding="utf-8",
    )
    # Deliberately DO NOT save gazetteer_path(4224): the cache is absent.

    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))

    assert len(env.romanization_instances) == 1
    gaz = env.romanization_instances[0]._gazetteer
    assert gaz is None, (
        f"expected no gazetteer applied with no cache file for the series; got {gaz!r}"
    )
    assert app.app_state.series_media_id == 4224, (
        "series must still be seeded from the pointer even when the gazetteer "
        "cache is absent"
    )
    assert app.romanization_pipeline.current_series_id() == 4224, (
        "the pipeline's series id must be stamped even when the cache is "
        "absent, so the hot loop doesn't reload every sentence"
    )


def test_build_pipeline_gazetteer_none_when_gazetteer_disabled(
    _pipeline_env, tmp_path,
):
    """Boot-seed, feature DISABLED: gazetteer.enabled: false -> gazetteer
    stays None AND no series is seeded, even with romanization enabled and a
    valid cached series pointer present -- the enabled flag gates the whole
    boot-restore, not just file presence.

    Seeded defect: in build_pipeline, drop the
    `if gaz_cfg.get("enabled", False)` guard on the `load_active_series()`
    call -> the gazetteer gets restored and the series gets seeded anyway ->
    the `gaz is None` / `series_media_id is None` assertions go red.
    """
    import json

    from yamabiko import paths
    from yamabiko.gazetteer.gazetteer import SeriesGazetteer, gazetteer_path

    env = _pipeline_env
    _write_overrides(tmp_path, {
        "romanization": {"enabled": True},
        "gazetteer": {"enabled": False},
    })
    paths.ensure_dirs()
    (paths.app_data_dir() / "active_series.json").write_text(
        json.dumps({"media_id": 4224, "label": "Test Series"}),
        encoding="utf-8",
    )
    SeriesGazetteer({"レックス": "Rex"}).save(gazetteer_path(4224))

    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))

    assert len(env.romanization_instances) == 1
    gaz = env.romanization_instances[0]._gazetteer
    assert gaz is None, (
        f"expected no gazetteer applied with gazetteer.enabled=False; got {gaz!r}"
    )
    assert app.app_state.series_media_id is None, (
        f"series must not be seeded when gazetteer disabled; got "
        f"{app.app_state.series_media_id!r}"
    )


async def _drive_run_pipeline_with_early_done(env, overrides):
    """Helper: drive build_pipeline + run_pipeline to completion with
    the done event pre-set, so run_pipeline exits through its normal
    cleanup path immediately after hotkey registration (and after
    task creation, but those tasks get cancelled in the finally
    block before they do any real work).

    Returns the PipelineApp with all run_pipeline side effects
    applied (hotkey_mgr set, manager set if browser enabled).
    """
    args = env.module.parse_args([])
    app = await env.module.build_pipeline(args)
    app.done.set()
    await env.module.run_pipeline(app)
    return app


def _stub_overlay_runtime(monkeypatch):
    """Stub the overlay subprocess + inner Job Object so run_pipeline tests
    don't spawn a real Qt overlay or touch real Windows Job Objects. The
    fake overlay proc stays 'alive' (wait raises TimeoutExpired) until it's
    terminated, so the overlay supervisor doesn't decide on its own that the
    overlay died."""
    import subprocess as _sp
    import yamabiko.lifecycle.jobs as _jobs_mod

    class _FakeJob:
        def __init__(self, name):
            self.name = name
            self.handle = None

        def assign(self, proc_or_pid):
            return True

        def close(self):
            pass

    monkeypatch.setattr(_jobs_mod, "create_inner_job", lambda name: _FakeJob(name))

    class _FakeOverlayProc:
        def __init__(self):
            self.pid = 999999
            self._handle = 0
            self.returncode = None

        def wait(self, timeout=None):
            if self.returncode is not None:
                return self.returncode
            raise _sp.TimeoutExpired(cmd="overlay", timeout=timeout)

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def poll(self):
            return self.returncode

    monkeypatch.setattr(_sp, "Popen", lambda *a, **k: _FakeOverlayProc())
    # Production routes the overlay spawn through the atomic
    # jobs.spawn_in_job seam (CREATE_SUSPENDED -> assign -> resume).
    # The real helper would feed the fake Popen's bogus pid into a real
    # Toolhelp32 thread walk (no threads found -> OSError -> the
    # supervisor counts spawn failures until give-up). Stub the seam to
    # hand back the same fake proc shape as the legacy path.
    monkeypatch.setattr(
        _jobs_mod, "spawn_in_job",
        lambda cmd, job, **kw: _FakeOverlayProc(),
    )


def test_run_pipeline_vad_chunker_crash_escalates(_pipeline_env, tmp_path, monkeypatch):
    """An exception in the VAD/chunker stage must be SURFACED via the
    escalate path (logs + sets app.done) instead of silently completing the
    task and exiting 0 with the exception unretrieved (no error, no watchdog
    restart, no dialog -- the app just vanishes).

    Seeded-defect drill: replace ``asyncio.create_task(_supervised_chunker())``
    with the bare ``app.chunker.run(...)`` -> the crash completes the task
    WITHOUT setting done -> app.done.is_set() is False -> this fails.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)

    async def _drive():
        args = env.module.parse_args([])
        app = await env.module.build_pipeline(args)

        class _CrashingChunker(_NoOpenWindow):
            async def run(self, in_q, out_q):
                raise RuntimeError("boom in Silero VAD")

        app.chunker = _CrashingChunker()
        await asyncio.wait_for(env.module.run_pipeline(app), timeout=15.0)
        return app

    app = asyncio.run(_drive())
    assert app.done.is_set(), (
        "VAD/chunker crash did not escalate (app.done not set); it exited "
        "silently instead of surfacing the failure"
    )


def test_run_pipeline_overlay_giveup_tears_down(_pipeline_env, tmp_path, monkeypatch):
    """When the overlay supervisor gives up (sets overlay_stop), the
    pipeline must tear down rather than run headless forever (ASR +
    translation burning CPU/GPU with no overlay, no tray, /livez green so
    the watchdog can't recover it).

    Seeded-defect drill: drop ``overlay_giveup_task`` from the wait_set ->
    setting overlay_stop no longer completes any awaited task -> run_pipeline
    hangs on the blocking stages -> wait_for times out -> this fails.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)

    async def _drive():
        args = env.module.parse_args([])
        app = await env.module.build_pipeline(args)

        class _BlockingChunker(_NoOpenWindow):
            async def run(self, in_q, out_q):
                await asyncio.Event().wait()  # never returns on its own

        app.chunker = _BlockingChunker()
        task = asyncio.create_task(env.module.run_pipeline(app))
        # Wait for run_pipeline to expose overlay_stop, then trip it.
        for _ in range(1000):
            if getattr(app, "overlay_stop", None) is not None:
                break
            await asyncio.sleep(0.01)
        assert getattr(app, "overlay_stop", None) is not None, (
            "run_pipeline never exposed overlay_stop"
        )
        assert not app.done.is_set()
        app.overlay_stop.set()  # simulate overlay supervisor give-up
        await asyncio.wait_for(task, timeout=15.0)

    asyncio.run(_drive())


def test_run_pipeline_teardown_logs_health_refresher_crash_at_error(
    _pipeline_env, tmp_path, caplog,
):
    """P2 (2026-07-12 review): the translator-health refresher logs
    nothing on its own death, and the teardown drain consumed the
    retrieved exception with a bare `_t.exception()` -- so a crashed
    refresher (stale /healthz data until exit) left ZERO evidence in the
    log. The drain must log a non-None retrieved exception at ERROR with
    the task's name. Exit-code semantics unchanged (run_pipeline still
    returns normally here).

    Seeded defect: revert the drain to `_t.exception()  # mark retrieved`
    -> no ERROR record naming the refresher task -> red.
    """
    import logging

    env = _pipeline_env
    _write_overrides(tmp_path)

    async def _drive():
        args = env.module.parse_args([])
        app = await env.module.build_pipeline(args)
        calls: list = []

        def _boom(reachable, detail=None):
            # Raises in the refresher's try AND again in its except
            # handler, so the second raise escapes the loop and kills the
            # task -- the exact silent-death shape under test.
            calls.append(reachable)
            raise RuntimeError("probe bookkeeping crashed")

        app.note_translator_health = _boom
        task = asyncio.create_task(env.module.run_pipeline(app))
        for _ in range(1000):
            if len(calls) >= 2:
                break
            await asyncio.sleep(0.01)
        assert len(calls) >= 2, "refresher never reached the crashing probe"
        # A couple of loop ticks so the escaped exception finalizes the task.
        for _ in range(5):
            await asyncio.sleep(0)
        app.done.set()
        await asyncio.wait_for(task, timeout=15.0)

    with caplog.at_level(logging.ERROR, logger="yamabiko"):
        asyncio.run(_drive())

    refresher_errors = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR
        and "translator-health-refresher" in r.getMessage()
    ]
    assert refresher_errors, (
        "the crashed health-refresher task was drained without an ERROR "
        "log naming it; its death is invisible in the field"
    )
    assert "probe bookkeeping crashed" in caplog.text, (
        "the retrieved exception's traceback/message is missing from the log"
    )


def test_run_pipeline_overlay_port_bind_failure_still_cleans_up(
    _pipeline_env, tmp_path, monkeypatch,
):
    """If the overlay server's serve() raises SystemExit (uvicorn calls
    sys.exit(1) on a port-bind failure, e.g. :8765 already in use), that
    SystemExit must NOT escape run_pipeline's finally and skip cleanup. The
    remaining teardown (capture.stop, translator close, etc.) must still run.

    Seeded-defect drill: narrow the finally's wait_for except back to
    (TimeoutError, CancelledError) only -> SystemExit propagates out of
    run_pipeline, capture.stop() never runs (resources leaked) -> this fails.
    """
    env = _pipeline_env
    _write_overrides(tmp_path, {"overlay": {"browser": True, "websocket_port": 8765}})

    import uvicorn

    class _BindFailServer:
        def __init__(self, config):
            self.config = config
            self.should_exit = False

        async def serve(self):
            raise SystemExit(1)  # uvicorn's behaviour on a failed port bind

    monkeypatch.setattr(uvicorn, "Server", _BindFailServer)

    async def _drive():
        args = env.module.parse_args([])
        app = await env.module.build_pipeline(args)

        class _BlockingChunker(_NoOpenWindow):
            async def run(self, in_q, out_q):
                await asyncio.Event().wait()

        app.chunker = _BlockingChunker()

        stopped = {"v": False}
        _orig_stop = app.capture.stop

        def _tracked_stop():
            stopped["v"] = True
            return _orig_stop()

        app.capture.stop = _tracked_stop
        # Must return normally (no SystemExit) and reach cleanup.
        await asyncio.wait_for(env.module.run_pipeline(app), timeout=15.0)
        return stopped["v"]

    stopped = asyncio.run(_drive())
    assert stopped, (
        "capture.stop() was not reached -- SystemExit from the overlay "
        "server escaped run_pipeline's finally and skipped cleanup"
    )


def test_run_pipeline_overlay_port_bind_failure_arms_startup_failure_exit_46(
    _pipeline_env, tmp_path, monkeypatch,
):
    """Through the REAL _serve_overlay catch site: when the overlay
    server's serve() raises SystemExit (uvicorn's behaviour on a
    port-bind failure, e.g. :8765 already in use), run_pipeline must arm
    ``startup_failure_requested`` so _exit_code_for returns
    EXIT_STARTUP_FAILURE (46). Without the armed flag the pipeline exits 0
    (STOP_SELF) and the watchdog silently cascades a full shutdown as if
    the user chose Quit -- the whole app disappears with no dialog.

    This drives the actual ``except (SystemExit, OSError)`` handler in
    _serve_overlay (not a stubbed _exit_code_for), so a regression that
    removes the arming there goes red here.

    Seeded-defect drill: delete ``app.startup_failure_requested = True``
    from _serve_overlay's except branch -> _exit_code_for(app) returns 0
    -> the ``== EXIT_STARTUP_FAILURE`` assertion goes red.
    """
    from yamabiko.lifecycle.exit_codes import EXIT_STARTUP_FAILURE

    env = _pipeline_env
    _write_overrides(tmp_path, {"overlay": {"browser": True, "websocket_port": 8765}})

    import uvicorn

    class _BindFailServer:
        def __init__(self, config):
            self.config = config
            self.should_exit = False

        async def serve(self):
            raise SystemExit(1)  # uvicorn's behaviour on a failed port bind

    monkeypatch.setattr(uvicorn, "Server", _BindFailServer)

    async def _drive():
        args = env.module.parse_args([])
        app = await env.module.build_pipeline(args)

        class _BlockingChunker(_NoOpenWindow):
            async def run(self, in_q, out_q):
                await asyncio.Event().wait()

        app.chunker = _BlockingChunker()
        await asyncio.wait_for(env.module.run_pipeline(app), timeout=15.0)
        return app

    app = asyncio.run(_drive())
    assert app.startup_failure_requested is True, (
        "port-bind SystemExit did not arm startup_failure_requested"
    )
    assert env.module._exit_code_for(app) == EXIT_STARTUP_FAILURE, (
        "a port-bind startup failure must map to EXIT_STARTUP_FAILURE "
        "(46); exit 0 would let the watchdog read it as a clean quit"
    )


def test_run_pipeline_overlay_giveup_arms_child_giveup_exit_47(
    _pipeline_env, tmp_path, monkeypatch,
):
    """When the overlay supervisor gives up (respawn budget exhausted),
    run_pipeline must arm ``overlay_giveup_requested`` so _exit_code_for
    returns EXIT_CHILD_GIVEUP (47, watchdog RESTART with a fresh overlay
    budget) -- NOT 0 (STOP_SELF), which would silently cascade a full
    shutdown of the whole app.

    The supervisor signals give-up by setting its ``gave_up`` flag before
    tripping overlay_stop; run_pipeline reads that flag after the wait to
    tell a flap-out from a clean overlay exit. We simulate the give-up by
    flipping the real supervisor's gave_up (exposed as
    app.overlay_supervisor) then tripping overlay_stop, exactly as the
    supervisor's cap branch does.

    Seeded-defect drill: delete the ``if overlay_supervisor.gave_up:
    app.overlay_giveup_requested = True`` block after run_pipeline's wait
    -> _exit_code_for(app) returns 0 -> the ``== EXIT_CHILD_GIVEUP``
    assertion goes red.
    """
    from yamabiko.lifecycle.exit_codes import EXIT_CHILD_GIVEUP

    env = _pipeline_env
    _write_overrides(tmp_path)

    async def _drive():
        args = env.module.parse_args([])
        app = await env.module.build_pipeline(args)

        class _BlockingChunker(_NoOpenWindow):
            async def run(self, in_q, out_q):
                await asyncio.Event().wait()

        app.chunker = _BlockingChunker()
        task = asyncio.create_task(env.module.run_pipeline(app))
        for _ in range(1000):
            if getattr(app, "overlay_supervisor", None) is not None:
                break
            await asyncio.sleep(0.01)
        assert getattr(app, "overlay_supervisor", None) is not None, (
            "run_pipeline never exposed overlay_supervisor"
        )
        # Simulate the supervisor's cap-branch give-up: gave_up set BEFORE
        # overlay_stop, exactly as PipelineSupervisor.run does.
        app.overlay_supervisor.gave_up = True
        app.overlay_stop.set()
        await asyncio.wait_for(task, timeout=15.0)
        return app

    app = asyncio.run(_drive())
    assert env.module._exit_code_for(app) == EXIT_CHILD_GIVEUP, (
        "an overlay respawn-budget give-up must map to EXIT_CHILD_GIVEUP "
        "(47); exit 0 would let the watchdog read it as a clean quit"
    )


def test_run_pipeline_normal_overlay_stop_does_not_arm_child_giveup(
    _pipeline_env, tmp_path, monkeypatch,
):
    """The 47 arming must be gated on the supervisor's gave_up flag, not
    on overlay_stop alone: overlay_stop is ALSO set on a clean overlay
    exit and by the pipeline's own teardown. A normal overlay_stop (no
    give-up) must leave overlay_giveup_requested False so _exit_code_for
    returns 0 (clean stop), not 47 (a spurious watchdog restart on every
    quit).

    Seeded-defect drill: change the arming guard from
    ``if overlay_supervisor.gave_up`` to an unconditional
    ``app.overlay_giveup_requested = True`` -> this test's clean-stop path
    returns 47 -> the ``== 0`` assertion goes red.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)

    async def _drive():
        args = env.module.parse_args([])
        app = await env.module.build_pipeline(args)

        class _BlockingChunker(_NoOpenWindow):
            async def run(self, in_q, out_q):
                await asyncio.Event().wait()

        app.chunker = _BlockingChunker()
        task = asyncio.create_task(env.module.run_pipeline(app))
        for _ in range(1000):
            if getattr(app, "overlay_supervisor", None) is not None:
                break
            await asyncio.sleep(0.01)
        assert getattr(app, "overlay_supervisor", None) is not None
        # Normal overlay_stop: NO give-up. gave_up stays False.
        assert app.overlay_supervisor.gave_up is False
        app.overlay_stop.set()
        await asyncio.wait_for(task, timeout=15.0)
        return app

    app = asyncio.run(_drive())
    assert app.overlay_giveup_requested is False, (
        "a normal overlay_stop wrongly armed overlay_giveup_requested"
    )
    assert env.module._exit_code_for(app) == 0, (
        "a clean overlay stop must map to 0 (STOP_SELF), not 47"
    )


def test_run_pipeline_supervisor_loop_crash_arms_child_giveup_exit_47(
    _pipeline_env, tmp_path, monkeypatch,
):
    """An UNEXPECTED exception inside the REAL PipelineSupervisor loop (a
    supervisor-code bug, e.g. a proc.wait raising something other than
    TimeoutExpired) must end in EXIT_CHILD_GIVEUP (47), not exit 0. The
    supervisor's defensive ``except Exception`` sets stop_event; without
    also marking gave_up, run_pipeline reads the stop as a clean quit and
    the watchdog silently cascades a full shutdown -- the same
    silent-death class as the cap give-up.

    Drives the REAL supervisor through run_pipeline: the fake overlay
    proc's ``wait()`` raises RuntimeError on its FIRST call (propagates
    out of the supervisor's inner wait loop, which only catches
    TimeoutExpired, to the defensive handler) and returns 0 afterwards
    (so the teardown ladder's stop() resolves cleanly).

    Seeded-defect drill: remove ``self.gave_up = True`` from the
    supervisor run()'s except branch -> exit code 0 -> red.
    """
    from yamabiko.lifecycle.exit_codes import EXIT_CHILD_GIVEUP

    env = _pipeline_env
    _write_overrides(tmp_path)

    import yamabiko.lifecycle.jobs as _jobs_mod

    class _WaitCrashProc:
        def __init__(self):
            self.pid = 999997
            self._handle = 0
            self.returncode = None
            self._crashed = False

        def wait(self, timeout=None):
            if not self._crashed:
                self._crashed = True
                raise RuntimeError("simulated supervisor-code bug")
            return 0

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        _jobs_mod, "spawn_in_job", lambda cmd, job, **kw: _WaitCrashProc(),
    )

    async def _drive():
        args = env.module.parse_args([])
        app = await env.module.build_pipeline(args)

        class _BlockingChunker(_NoOpenWindow):
            async def run(self, in_q, out_q):
                await asyncio.Event().wait()

        app.chunker = _BlockingChunker()
        await asyncio.wait_for(env.module.run_pipeline(app), timeout=15.0)
        return app

    app = asyncio.run(_drive())
    assert app.overlay_giveup_requested is True, (
        "a supervisor-loop crash did not arm overlay_giveup_requested"
    )
    assert env.module._exit_code_for(app) == EXIT_CHILD_GIVEUP, (
        "a supervisor-loop crash must map to EXIT_CHILD_GIVEUP (47); exit "
        "0 would let the watchdog read it as a clean quit and silently "
        "cascade a full shutdown"
    )


def test_run_pipeline_real_cap_exhaustion_arms_child_giveup_exit_47(
    _pipeline_env, tmp_path, monkeypatch,
):
    """End-to-end across the supervisor -> run_pipeline seam with NO flag
    faking: overlay procs crash (rc=1 -> classify RESTART) until the REAL
    PipelineSupervisor's respawn budget (max_restarts=1 via overrides)
    exhausts, its REAL cap branch sets gave_up + overlay_stop, and
    run_pipeline arms overlay_giveup_requested -> _exit_code_for == 47.

    Documents-existing-behavior note (charter Iron Law q2): the sibling
    tests cover each side of this seam separately (the supervisor cap
    branch with fakes; run_pipeline's arming with a directly-flipped
    gave_up), so this test is green at introduction BY DESIGN -- its value
    is failing when the two sides drift apart. Seeded-defect drill RUN at
    introduction: removing ``self.gave_up = True`` from the cap branch
    went red here (and only here among run_pipeline tests).

    Real time cost: one backoff tier (~1 s, backoff_s[0]) between the
    initial spawn and the single budgeted respawn.

    Seeded defects (one line each):
      - remove ``self.gave_up = True`` from the supervisor cap branch ->
        run_pipeline reads False -> exit 0 -> red.
      - break run_pipeline's post-wait read (drop the arming block) ->
        exit 0 -> red.
    """
    from yamabiko.lifecycle.exit_codes import EXIT_CHILD_GIVEUP

    env = _pipeline_env
    _write_overrides(tmp_path, {"lifecycle": {"overlay": {"max_restarts": 1}}})

    import yamabiko.lifecycle.jobs as _jobs_mod

    class _CrashingProc:
        def __init__(self):
            self.pid = 999996
            self._handle = 0
            self.returncode = 1

        def wait(self, timeout=None):
            return 1  # crash exit -> classify RESTART -> respawn loop

        def terminate(self):
            pass

        def kill(self):
            pass

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        _jobs_mod, "spawn_in_job", lambda cmd, job, **kw: _CrashingProc(),
    )

    async def _drive():
        args = env.module.parse_args([])
        app = await env.module.build_pipeline(args)

        class _BlockingChunker(_NoOpenWindow):
            async def run(self, in_q, out_q):
                await asyncio.Event().wait()

        app.chunker = _BlockingChunker()
        await asyncio.wait_for(env.module.run_pipeline(app), timeout=15.0)
        return app

    app = asyncio.run(_drive())
    assert app.overlay_supervisor.gave_up is True, (
        "the REAL supervisor cap branch never set gave_up despite the "
        "overlay crash-looping past max_restarts=1"
    )
    assert app.overlay_giveup_requested is True, (
        "run_pipeline did not arm overlay_giveup_requested from the REAL "
        "supervisor's give-up"
    )
    assert env.module._exit_code_for(app) == EXIT_CHILD_GIVEUP, (
        "a real cap exhaustion must map to EXIT_CHILD_GIVEUP (47)"
    )


def test_run_pipeline_wires_hotkey_manager_when_enabled(_pipeline_env, tmp_path):
    """When cfg['hotkeys']['enabled'] is True, run_pipeline must
    construct a HotkeyManager, call .register() with callable
    on_pause / on_toggle_overlay / on_conflict handlers and the
    configured keybindings, and stash the instance on
    PipelineApp.hotkey_mgr.

    Hotkey wiring lives in run_pipeline (not build_pipeline) because
    its closures reference app.manager which is populated by the
    browser-server branch. A wiring regression (keybindings not
    forwarded, callbacks missing, hotkey_mgr never assigned) would
    ship without a test covering this path.

    Seeded-defect check: in run_pipeline, change
        app.hotkey_mgr = hk_mod.HotkeyManager()
    to
        app.hotkey_mgr = None  # forget
    -> identity assertion fails because the fake registered but
    wasn't stored on the app.
    """
    env = _pipeline_env
    _write_overrides(tmp_path, {
        "hotkeys": {
            "enabled": True,
            "pause_resume": "Ctrl+Alt+P",
            "toggle_overlay": "Ctrl+Alt+O",
            "show_series": "Ctrl+Alt+S",
        },
        # Pin the series hotkey feature ON so the show_series assertion below
        # is deterministic. feature_enabled requires BOTH flags, so
        # romanization must be on too or the gate would drop the combo
        # (that romanization-off path is the paired divergence test below).
        "gazetteer": {"enabled": True},
        "romanization": {"enabled": True},
    })

    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert app.hotkey_mgr is env.fake_hotkey_mgr, (
        "app.hotkey_mgr is not the HotkeyManager instance the "
        "fixture injected; run_pipeline didn't wire through the "
        "monkeypatched constructor"
    )
    assert len(env.fake_hotkey_mgr.register_calls) == 1, (
        f"expected exactly one register() call; got "
        f"{len(env.fake_hotkey_mgr.register_calls)}"
    )
    call = env.fake_hotkey_mgr.register_calls[0]
    assert call["pause"] == "Ctrl+Alt+P", (
        f"cfg pause_resume keybind did not reach register(); "
        f"got {call['pause']!r}"
    )
    assert call["toggle_overlay"] == "Ctrl+Alt+O", (
        f"cfg toggle_overlay keybind did not reach register(); "
        f"got {call['toggle_overlay']!r}"
    )
    assert call["show_series"] == "Ctrl+Alt+S", (
        f"cfg show_series keybind did not reach register(); "
        f"got {call.get('show_series')!r}"
    )
    assert callable(call["on_pause"]), "on_pause callback not callable"
    assert callable(call["on_toggle_overlay"]), (
        "on_toggle_overlay callback not callable"
    )
    assert callable(call["on_show_series"]), (
        "on_show_series callback not callable"
    )
    assert callable(call["on_conflict"]), "on_conflict callback not callable"


def test_run_pipeline_skips_hotkey_manager_when_disabled(_pipeline_env, tmp_path):
    """Paired negative case: hotkeys.enabled=False means no
    HotkeyManager is constructed and app.hotkey_mgr stays None.

    Seeded-defect check: remove the
        `if hk_cfg.get("enabled", True):`
    guard in run_pipeline -> HotkeyManager is constructed regardless,
    register_calls has length 1, this assertion fails.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)  # fixture default disables hotkeys

    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert app.hotkey_mgr is None, (
        f"hotkey_mgr must be None when hotkeys disabled; "
        f"got {app.hotkey_mgr!r}"
    )
    assert len(env.fake_hotkey_mgr.register_calls) == 0, (
        f"HotkeyManager.register was called {len(env.fake_hotkey_mgr.register_calls)} "
        f"time(s) even though hotkeys were disabled"
    )


def test_run_pipeline_binds_no_series_hotkey_when_gazetteer_disabled(
    _pipeline_env, tmp_path,
):
    """The series-palette hotkey is part of the opt-in gazetteer feature.
    With hotkeys.enabled=True but gazetteer.enabled=False, run_pipeline must
    still register the pause/toggle_overlay combos but pass an EMPTY
    show_series combo, so HotkeyManager.register (which skips any falsy combo)
    binds no series hotkey. This is what makes gazetteer.enabled=false
    genuinely disable the feature, not merely skip the boot-restore.

    Seeded-defect drill: revert the show_series kwarg in run_pipeline to the
    unconditional `hk_cfg.get("show_series", "Ctrl+Alt+S")` (flag ignored) ->
    the combo "Ctrl+Alt+S" is passed even when disabled -> the show_series==""
    assertion goes red.
    """
    env = _pipeline_env
    _write_overrides(tmp_path, {
        "hotkeys": {
            "enabled": True,
            "pause_resume": "Ctrl+Alt+P",
            "toggle_overlay": "Ctrl+Alt+O",
            "show_series": "Ctrl+Alt+S",
        },
        "gazetteer": {"enabled": False},
    })

    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert app.hotkey_mgr is env.fake_hotkey_mgr
    assert len(env.fake_hotkey_mgr.register_calls) == 1, (
        f"expected exactly one register() call; got "
        f"{len(env.fake_hotkey_mgr.register_calls)}"
    )
    call = env.fake_hotkey_mgr.register_calls[0]
    assert call["show_series"] == "", (
        f"gazetteer.enabled=False must pass an EMPTY show_series combo so no "
        f"series hotkey binds; got {call.get('show_series')!r}"
    )
    # The other hotkeys are unaffected by the gazetteer flag.
    assert call["pause"] == "Ctrl+Alt+P", (
        f"disabling the gazetteer must not disturb the pause combo; "
        f"got {call.get('pause')!r}"
    )
    assert call["toggle_overlay"] == "Ctrl+Alt+O", (
        f"disabling the gazetteer must not disturb the toggle_overlay combo; "
        f"got {call.get('toggle_overlay')!r}"
    )


def test_series_hotkey_not_bound_when_romanization_disabled(
    _pipeline_env, tmp_path,
):
    """Flag divergence: gazetteer.enabled=True but romanization.enabled=False.
    A series hint can only reach the model through RomanizationPipeline, which
    is never built when romanization is off, so the series hotkey must NOT
    bind -- run_pipeline gates it on feature_enabled (both flags), not the raw
    gazetteer flag. The pause/toggle combos are unaffected.

    Seeded-defect drill: revert run_pipeline's `_gaz_enabled =
    feature_enabled(app.cfg)` to the raw
    `app.cfg.get("gazetteer", {}).get("enabled", False)` -> gazetteer.enabled
    alone binds "Ctrl+Alt+S" here -> the show_series=="" assertion goes red.
    """
    env = _pipeline_env
    _write_overrides(tmp_path, {
        "hotkeys": {
            "enabled": True,
            "pause_resume": "Ctrl+Alt+P",
            "toggle_overlay": "Ctrl+Alt+O",
            "show_series": "Ctrl+Alt+S",
        },
        "gazetteer": {"enabled": True},
        "romanization": {"enabled": False},
    })

    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert app.hotkey_mgr is env.fake_hotkey_mgr
    call = env.fake_hotkey_mgr.register_calls[0]
    assert call["show_series"] == "", (
        f"gazetteer on but romanization off must bind NO series hotkey (the "
        f"hint would have no pipeline to flow through); got "
        f"{call.get('show_series')!r}"
    )
    assert call["pause"] == "Ctrl+Alt+P"
    assert call["toggle_overlay"] == "Ctrl+Alt+O"


def test_amain_returns_43_on_user_quit(monkeypatch):
    """amain maps a USER quit (stop_all_requested) to STOP_ALL exit code 43 so
    the watchdog stops -- even if run_pipeline's teardown then raised -- while a
    genuine crash with no quit intent still propagates to a non-zero exit the
    watchdog restarts. Without this, the overlay quit's slow teardown exits with a
    code the watchdog reads as a crash and restarts, three times, to give-up.

    Seeded defect (1): drop the `if app.stop_all_requested: return 43` branches in
    amain -> the quit cases return 0 / re-raise -> red.
    Seeded defect (2): narrow amain's `except BaseException` back to
    `except Exception` -> the CancelledError / SystemExit quit cases (the actual
    dirty-teardown kinds from the beta loop -- both are BaseException, not
    Exception) re-raise instead of returning 43 -> red.
    """
    import yamabiko.main as _main_mod

    class _App:
        def __init__(self):
            self.stop_all_requested = False
            self.stream_death_requested = False
            self.stage_escalation_requested = False

    app = _App()

    async def _build(_args):
        return app
    monkeypatch.setattr(_main_mod, "build_pipeline", _build)

    # Clean run, no quit -> 0
    async def _run_clean(_a):
        return None
    monkeypatch.setattr(_main_mod, "run_pipeline", _run_clean)
    app.stop_all_requested = False
    assert asyncio.run(_main_mod.amain(object())) == 0

    # User quit, clean teardown -> 43
    async def _run_quit(a):
        a.stop_all_requested = True
    monkeypatch.setattr(_main_mod, "run_pipeline", _run_quit)
    app.stop_all_requested = False
    assert asyncio.run(_main_mod.amain(object())) == 43

    # User quit, teardown RAISED -> STILL 43 (must not become a watchdog restart)
    async def _run_quit_raise(a):
        a.stop_all_requested = True
        raise RuntimeError("messy teardown after quit")
    monkeypatch.setattr(_main_mod, "run_pipeline", _run_quit_raise)
    app.stop_all_requested = False
    assert asyncio.run(_main_mod.amain(object())) == 43

    # User quit, teardown raised asyncio.CancelledError -> STILL 43. CancelledError
    # is a BaseException (since 3.8), raised by task cancellation during shutdown;
    # under a narrow `except Exception` it escaped -> exit 1 -> watchdog restart.
    async def _run_quit_cancelled(a):
        a.stop_all_requested = True
        raise asyncio.CancelledError()
    monkeypatch.setattr(_main_mod, "run_pipeline", _run_quit_cancelled)
    app.stop_all_requested = False
    assert asyncio.run(_main_mod.amain(object())) == 43

    # User quit, teardown raised SystemExit (uvicorn's bind/shutdown path) -> 43.
    async def _run_quit_sysexit(a):
        a.stop_all_requested = True
        raise SystemExit(1)
    monkeypatch.setattr(_main_mod, "run_pipeline", _run_quit_sysexit)
    app.stop_all_requested = False
    assert asyncio.run(_main_mod.amain(object())) == 43

    # KeyboardInterrupt is NOT a tray/overlay quit: it must propagate even with
    # stop_all set, never be coerced to 43 (amain re-raises it ahead of the
    # BaseException branch). Seeded defect: drop amain's `except KeyboardInterrupt:
    # raise` -> this returns 43 instead of propagating -> red.
    async def _run_quit_kbint(a):
        a.stop_all_requested = True
        raise KeyboardInterrupt()
    monkeypatch.setattr(_main_mod, "run_pipeline", _run_quit_kbint)
    app.stop_all_requested = False
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(_main_mod.amain(object()))

    # Genuine crash, NO quit -> propagates (non-zero exit -> watchdog restart)
    async def _run_crash(_a):
        raise RuntimeError("real pipeline crash")
    monkeypatch.setattr(_main_mod, "run_pipeline", _run_crash)
    app.stop_all_requested = False
    with pytest.raises(RuntimeError, match="real pipeline crash"):
        asyncio.run(_main_mod.amain(object()))


class _HardExitCalled(BaseException):
    """Sentinel: stands in for the real os._exit so tests can observe the code
    without terminating the pytest process."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _patch_main_preamble(monkeypatch, mod):
    """Stub parse_args / log setup / wizard gate AND os._exit so main() runs
    in-process. Returns a dict whose 'hard_exit_code' is set iff main() hard-exits
    (os._exit), so a real hard-exit is observable instead of killing the runner."""
    from types import SimpleNamespace
    monkeypatch.setattr(mod, "parse_args", lambda: SimpleNamespace())
    monkeypatch.setattr(mod.log_config, "setup_main_log", lambda *a, **k: None)
    monkeypatch.setattr(mod.wizard_state, "needs_wizard", lambda: False)
    captured: dict = {}

    def _fake_os_exit(code):
        captured["hard_exit_code"] = code
        raise _HardExitCalled(code)
    monkeypatch.setattr(mod.os, "_exit", _fake_os_exit)
    return captured


def test_main_hard_exits_43_on_clean_quit(monkeypatch):
    """A user quit (amain returns 43) must HARD-exit 43 via os._exit, NOT sys.exit:
    a normal interpreter shutdown joins native-extension threads (onnxruntime /
    CTranslate2 / WASAPI) whose finalization mangles the OS exit code, so the
    watchdog reads a non-43 code and restart-loops. os._exit delivers 43
    verbatim.

    Seeded defect: change main()'s `_hard_exit(code)` back to `sys.exit(code)` ->
    os._exit never called -> 'hard_exit_code' absent -> red.
    """
    import yamabiko.main as _m
    cap = _patch_main_preamble(monkeypatch, _m)

    class _App:
        stop_all_requested = True

    async def _fake_amain(args, app_box=None):
        app_box["app"] = _App()
        return 43
    monkeypatch.setattr(_m, "amain", _fake_amain)

    with pytest.raises(_HardExitCalled):
        _m.main()
    assert cap.get("hard_exit_code") == 43, f"quit must os._exit(43); got {cap}"


def test_main_hard_exits_43_when_asyncio_run_cleanup_raises_after_quit(monkeypatch):
    """Same STOP_ALL(43) guarantee when asyncio.run's OWN post-amain cleanup raises
    a BaseException after the quit -- it's OUTSIDE amain, so main()'s handler must
    hard-exit 43 (else it escapes as a crash exit -> watchdog restart loop).

    Seeded defect: narrow main()'s `except BaseException` to KeyboardInterrupt ->
    SystemExit propagates, no os._exit(43) -> red.
    """
    import yamabiko.main as _m
    cap = _patch_main_preamble(monkeypatch, _m)

    class _App:
        stop_all_requested = True

    async def _fake_amain(args, app_box=None):
        app_box["app"] = _App()           # quit recorded before cleanup blows up
        raise SystemExit(1)               # simulate asyncio.run cleanup raising
    monkeypatch.setattr(_m, "amain", _fake_amain)

    with pytest.raises(_HardExitCalled):
        _m.main()
    assert cap.get("hard_exit_code") == 43, f"quit must os._exit(43); got {cap}"


def test_main_propagates_crash_without_quit(monkeypatch):
    """A genuine crash (asyncio.run raises, NO quit intent) must propagate, NOT
    become a hard-exit 43 -- the watchdog should restart a real crash.

    Seeded defect: drop the `stop_all_requested` guard and always _hard_exit(43)
    -> a real crash silently becomes STOP_ALL -> red (RuntimeError not raised).
    """
    import yamabiko.main as _m
    cap = _patch_main_preamble(monkeypatch, _m)

    class _App:
        stop_all_requested = False

    async def _fake_amain(args, app_box=None):
        app_box["app"] = _App()
        raise RuntimeError("genuine crash, no quit")
    monkeypatch.setattr(_m, "amain", _fake_amain)

    with pytest.raises(RuntimeError, match="genuine crash, no quit"):
        _m.main()
    assert "hard_exit_code" not in cap, "a real crash must NOT hard-exit 43"


def test_amain_returns_stream_death_code_when_teardown_raises(monkeypatch):
    """amain's EXCEPT path (run_pipeline raised during teardown) must honor a
    stream-death the same way it honors a user quit: a capture death whose
    teardown then raised must STILL return EXIT_STREAM_DEATH (RESTART); and if a
    user quit RACES it, STOP_ALL(43) must win. The clean-path ternary is covered
    by the other amain tests; this pins the except-branch precedence -- a
    one-line reorder there would invert RESTART/STOP while the suite stayed
    green.

    Seeded defect (1): drop the `if app.stream_death_requested: return
    EXIT_STREAM_DEATH` branch in amain's except -> a capture death with a raised
    teardown re-raises instead of returning 44 -> red.
    Seeded defect (2): move the stream_death check before the stop_all check in
    the except -> the racing case returns 44 instead of 43 -> red.
    """
    import yamabiko.main as _main_mod
    from yamabiko.lifecycle.exit_codes import EXIT_STREAM_DEATH

    class _App:
        def __init__(self):
            self.stop_all_requested = False
            self.stream_death_requested = False

    app = _App()

    async def _build(_args):
        return app
    monkeypatch.setattr(_main_mod, "build_pipeline", _build)

    # Capture death, teardown RAISED -> EXIT_STREAM_DEATH (RESTART)
    async def _run_death_raise(a):
        a.stream_death_requested = True
        raise RuntimeError("messy teardown after capture death")
    monkeypatch.setattr(_main_mod, "run_pipeline", _run_death_raise)
    app.stop_all_requested = False
    app.stream_death_requested = False
    assert asyncio.run(_main_mod.amain(object())) == EXIT_STREAM_DEATH

    # User quit RACES the capture death in teardown -> STOP_ALL(43) wins
    async def _run_both_raise(a):
        a.stream_death_requested = True
        a.stop_all_requested = True
        raise RuntimeError("quit + capture death in teardown")
    monkeypatch.setattr(_main_mod, "run_pipeline", _run_both_raise)
    app.stop_all_requested = False
    app.stream_death_requested = False
    assert asyncio.run(_main_mod.amain(object())) == 43


def test_amain_returns_stage_escalation_code_when_teardown_raises(monkeypatch):
    """amain's EXCEPT path must honor a stage escalation the same way it
    honors a capture death: an escalation whose teardown then raised must
    STILL return EXIT_STAGE_ESCALATION (RESTART) -- not re-raise into a
    generic crash exit -- and both stop_all(43) and stream_death(44) must
    outrank it if they race on the same teardown.

    Seeded defect (1): drop the stage_escalation branch from amain's
    except -> an escalation with a raised teardown re-raises instead of
    returning 45 -> red.
    Seeded defect (2): move the stage_escalation check above the stop_all
    check in the except -> the quit race returns 45 instead of 43 -> red.
    """
    import yamabiko.main as _main_mod
    from yamabiko.lifecycle.exit_codes import (
        EXIT_STAGE_ESCALATION, EXIT_STREAM_DEATH,
    )

    class _App:
        def __init__(self):
            self.stop_all_requested = False
            self.stream_death_requested = False
            self.stage_escalation_requested = False

    app = _App()

    async def _build(_args):
        return app
    monkeypatch.setattr(_main_mod, "build_pipeline", _build)

    def _reset():
        app.stop_all_requested = False
        app.stream_death_requested = False
        app.stage_escalation_requested = False

    # Stage escalation, teardown RAISED -> EXIT_STAGE_ESCALATION (RESTART)
    async def _run_escalate_raise(a):
        a.stage_escalation_requested = True
        raise RuntimeError("messy teardown after stage escalation")
    monkeypatch.setattr(_main_mod, "run_pipeline", _run_escalate_raise)
    _reset()
    assert asyncio.run(_main_mod.amain(object())) == EXIT_STAGE_ESCALATION

    # User quit RACES the escalation in teardown -> STOP_ALL(43) wins
    async def _run_quit_and_escalate_raise(a):
        a.stage_escalation_requested = True
        a.stop_all_requested = True
        raise RuntimeError("quit + stage escalation in teardown")
    monkeypatch.setattr(_main_mod, "run_pipeline", _run_quit_and_escalate_raise)
    _reset()
    assert asyncio.run(_main_mod.amain(object())) == 43

    # Capture death RACES the escalation -> stream death (44) outranks 45
    async def _run_death_and_escalate_raise(a):
        a.stage_escalation_requested = True
        a.stream_death_requested = True
        raise RuntimeError("capture death + stage escalation in teardown")
    monkeypatch.setattr(_main_mod, "run_pipeline", _run_death_and_escalate_raise)
    _reset()
    assert asyncio.run(_main_mod.amain(object())) == EXIT_STREAM_DEATH


def test_amain_returns_startup_and_giveup_codes_when_teardown_raises(monkeypatch):
    """amain's EXCEPT path (run_pipeline raised during teardown) must honor
    the two new typed shutdowns: an overlay port-bind startup failure
    returns EXIT_STARTUP_FAILURE (46), and an overlay respawn give-up
    returns EXIT_CHILD_GIVEUP (47) -- and this path also logs via
    ``_INTENT_LABELS[_code]``, so a missing label entry would KeyError
    here instead of naming the intent.

    Seeded defect (1): drop the startup_failure branch from _exit_code_for
    -> the port-bind teardown re-raises instead of returning 46 -> red.
    Seeded defect (2): omit 46/47 from _INTENT_LABELS -> KeyError in the
    except branch's log line -> red (not a clean return).
    """
    import yamabiko.main as _main_mod
    from yamabiko.lifecycle.exit_codes import (
        EXIT_STARTUP_FAILURE, EXIT_CHILD_GIVEUP,
    )

    class _App:
        def __init__(self):
            self.stop_all_requested = False
            self.stream_death_requested = False
            self.stage_escalation_requested = False
            self.startup_failure_requested = False
            self.overlay_giveup_requested = False

    app = _App()

    async def _build(_args):
        return app
    monkeypatch.setattr(_main_mod, "build_pipeline", _build)

    # Overlay port-bind failure, teardown RAISED -> EXIT_STARTUP_FAILURE.
    async def _run_startup_fail_raise(a):
        a.startup_failure_requested = True
        raise RuntimeError("messy teardown after port-bind failure")
    monkeypatch.setattr(_main_mod, "run_pipeline", _run_startup_fail_raise)
    app.startup_failure_requested = False
    app.overlay_giveup_requested = False
    assert asyncio.run(_main_mod.amain(object())) == EXIT_STARTUP_FAILURE

    # Overlay give-up, teardown RAISED -> EXIT_CHILD_GIVEUP.
    async def _run_giveup_raise(a):
        a.overlay_giveup_requested = True
        raise RuntimeError("messy teardown after overlay give-up")
    monkeypatch.setattr(_main_mod, "run_pipeline", _run_giveup_raise)
    app.startup_failure_requested = False
    app.overlay_giveup_requested = False
    assert asyncio.run(_main_mod.amain(object())) == EXIT_CHILD_GIVEUP


def test_main_hard_exits_stream_death_when_asyncio_cleanup_raises(monkeypatch):
    """When asyncio.run's OWN post-amain cleanup raises after a CAPTURE DEATH
    (stream_death_requested), main() must _hard_exit(EXIT_STREAM_DEATH) so the
    watchdog sees 44 (RESTART), mirroring the stop_all(43) guarantee. Without
    it the cleanup-raise escapes as exit 1 -- mangled by native-thread
    finalization, the same root cause _hard_exit/os._exit exist to prevent --
    and the device is never re-resolved.

    Seeded defect: drop the stream_death branch in main()'s except -> a
    capture-death-with-leaked-cleanup re-raises instead of hard-exiting 44 -> red.
    """
    import yamabiko.main as _m
    from yamabiko.lifecycle.exit_codes import EXIT_STREAM_DEATH
    cap = _patch_main_preamble(monkeypatch, _m)

    class _App:
        stop_all_requested = False
        stream_death_requested = True

    async def _fake_amain(args, app_box=None):
        app_box["app"] = _App()      # death recorded before cleanup blows up
        raise SystemExit(1)          # simulate asyncio.run cleanup raising
    monkeypatch.setattr(_m, "amain", _fake_amain)

    with pytest.raises(_HardExitCalled):
        _m.main()
    assert cap.get("hard_exit_code") == EXIT_STREAM_DEATH, (
        f"a capture death whose asyncio cleanup raised must os._exit(44); got {cap}"
    )


def test_main_hard_exits_stage_escalation_when_asyncio_cleanup_raises(monkeypatch):
    """When asyncio.run's OWN post-amain cleanup raises after a STAGE
    ESCALATION (stage_escalation_requested), main() must
    _hard_exit(EXIT_STAGE_ESCALATION) so the watchdog sees 45 (RESTART),
    mirroring the stream-death(44) guarantee. os._exit is the unmangled
    delivery channel: interpreter finalization joins native-extension
    threads (onnxruntime / CTranslate2 / WASAPI) that can mangle the code
    en route to the watchdog.

    Seeded defect: drop the stage_escalation branch in main()'s except ->
    an escalation-with-leaked-cleanup re-raises instead of hard-exiting 45
    -> _HardExitCalled not raised -> red.
    """
    import yamabiko.main as _m
    from yamabiko.lifecycle.exit_codes import EXIT_STAGE_ESCALATION
    cap = _patch_main_preamble(monkeypatch, _m)

    class _App:
        stop_all_requested = False
        stream_death_requested = False
        stage_escalation_requested = True

    async def _fake_amain(args, app_box=None):
        app_box["app"] = _App()      # escalation recorded before cleanup blows up
        raise SystemExit(1)          # simulate asyncio.run cleanup raising
    monkeypatch.setattr(_m, "amain", _fake_amain)

    with pytest.raises(_HardExitCalled):
        _m.main()
    assert cap.get("hard_exit_code") == EXIT_STAGE_ESCALATION, (
        f"a stage escalation whose asyncio cleanup raised must os._exit(45); "
        f"got {cap}"
    )


def test_exit_code_for_precedence_stop_all_stream_death_stage_escalation():
    """_exit_code_for maps a user quit to 43, a capture death to
    EXIT_STREAM_DEATH, a stage escalation to EXIT_STAGE_ESCALATION, and a
    clean stop to 0 -- with precedence stop_all(43) > stream_death(44) >
    stage_escalation(45) > 0. A user quit RACING either recoverable
    failure must STOP (43), never RESTART. amain and the teardown
    hard-exit both read the code from here, so the precedence lives in one
    place.

    Seeded defects (one line each):
      - flip the stream_death/stop_all precedence -> a quit that raced a
        stream death returns 44 (restart) -> red.
      - drop the stage_escalation branch (revert to `... else 0`) -> a
        stage escalation returns 0 (STOP_SELF: the watchdog reads a stage
        crash-loop as a clean quit and silently stops everything) -> red.
      - reorder stage_escalation above stream_death -> the death+escalation
        race returns 45 instead of 44 -> red.
    """
    import yamabiko.main as _m
    from yamabiko.lifecycle.exit_codes import (
        EXIT_STAGE_ESCALATION, EXIT_STREAM_DEATH,
    )

    class _App:
        def __init__(self, stop_all, stream_death, stage_escalation=False):
            self.stop_all_requested = stop_all
            self.stream_death_requested = stream_death
            self.stage_escalation_requested = stage_escalation

    assert _m._exit_code_for(_App(True, False)) == 43
    assert _m._exit_code_for(_App(False, True)) == EXIT_STREAM_DEATH
    assert _m._exit_code_for(_App(False, False)) == 0
    assert _m._exit_code_for(_App(True, True)) == 43  # quit beats race
    # Stage escalation alone -> 45 (RESTART), never 0 (STOP_SELF).
    assert _m._exit_code_for(_App(False, False, True)) == EXIT_STAGE_ESCALATION
    # A user quit racing a stage escalation must STOP.
    assert _m._exit_code_for(_App(True, False, True)) == 43
    # A capture death racing a stage escalation: stream death outranks
    # escalation (both RESTART, but 44 carries the device-specific
    # give-up copy in the watchdog).
    assert _m._exit_code_for(_App(False, True, True)) == EXIT_STREAM_DEATH
    # All three race: quit still wins.
    assert _m._exit_code_for(_App(True, True, True)) == 43


def test_exit_code_for_startup_failure_and_overlay_giveup_precedence():
    """_exit_code_for maps an overlay-server port-bind failure to
    EXIT_STARTUP_FAILURE (46) and an overlay respawn-budget exhaustion to
    EXIT_CHILD_GIVEUP (47), with precedence:

        stop_all(43) > startup_failure(46) > stream_death(44) >
        stage_escalation(45) > overlay_giveup(47) > 0.

    A fatal 'can't even bind the overlay port' (46) outranks the
    recoverable RESTART codes (44/45) -- a bind conflict won't self-heal,
    so surfacing the port dialog beats a silent restart. overlay_giveup
    (47) is the lowest-priority nonzero code: it is itself a RESTART, and
    any more-specific reason should win. A user quit (stop_all) still
    wins over both new codes -- a deliberate quit needs no error dialog.

    Seeded defects (one line each in _exit_code_for):
      - move the startup_failure branch BELOW stream_death -> the
        startup_failure+stream_death race returns 44 -> red.
      - drop the startup_failure branch -> a port-bind failure returns 0
        (STOP_SELF: the silent-death bug) -> red.
      - move the overlay_giveup branch ABOVE stage_escalation -> the
        escalation+giveup race returns 47 -> red.
    """
    import yamabiko.main as _m
    from yamabiko.lifecycle.exit_codes import (
        EXIT_STARTUP_FAILURE, EXIT_CHILD_GIVEUP, EXIT_STREAM_DEATH,
        EXIT_STAGE_ESCALATION,
    )

    class _App:
        def __init__(self, stop_all=False, startup_failure=False,
                     stream_death=False, stage_escalation=False,
                     overlay_giveup=False):
            self.stop_all_requested = stop_all
            self.startup_failure_requested = startup_failure
            self.stream_death_requested = stream_death
            self.stage_escalation_requested = stage_escalation
            self.overlay_giveup_requested = overlay_giveup

    # Each new code alone.
    assert _m._exit_code_for(_App(startup_failure=True)) == EXIT_STARTUP_FAILURE
    assert _m._exit_code_for(_App(overlay_giveup=True)) == EXIT_CHILD_GIVEUP
    # startup_failure (fatal, no restart) outranks the recoverable codes.
    assert _m._exit_code_for(
        _App(startup_failure=True, stream_death=True)
    ) == EXIT_STARTUP_FAILURE
    assert _m._exit_code_for(
        _App(startup_failure=True, stage_escalation=True)
    ) == EXIT_STARTUP_FAILURE
    # A user quit still wins over a startup failure.
    assert _m._exit_code_for(_App(stop_all=True, startup_failure=True)) == 43
    # overlay_giveup is lowest: a racing stage escalation wins.
    assert _m._exit_code_for(
        _App(stage_escalation=True, overlay_giveup=True)
    ) == EXIT_STAGE_ESCALATION
    # A user quit wins over an overlay give-up.
    assert _m._exit_code_for(_App(stop_all=True, overlay_giveup=True)) == 43


def test_stop_capture_bounded_hard_exits_when_capture_stop_wedges(monkeypatch):
    """A wedged capture.stop (stuck WASAPI/PortAudio driver) must trigger an
    immediate _hard_exit, NOT a log-and-continue. The to_thread worker runs on
    asyncio's DEFAULT executor; if teardown merely abandons it, asyncio.run's
    Runner.close() re-joins it via shutdown_default_executor(THREAD_JOIN_TIMEOUT
    == 300 s on 3.13) BEFORE main() reaches its own _hard_exit -- so a tray-Quit
    on a stuck driver hangs ~5 min (issue #55). _stop_capture_bounded must
    os._exit from inside the timeout branch with the correct code (here 43).

    Seeded defect: revert the TimeoutError branch to log-and-continue ->
    _stop_capture_bounded returns without calling _hard_exit -> DID NOT RAISE -> red.
    """
    import asyncio
    import threading
    import yamabiko.main as _m

    cap: dict = {}

    def _fake_hard_exit(code):
        cap["code"] = code
        raise _HardExitCalled(code)
    monkeypatch.setattr(_m, "_hard_exit", _fake_hard_exit)

    released = threading.Event()

    class _Cap:
        def stop(self):
            released.wait(timeout=30)  # the "wedge": blocks until released

    class _App:
        capture = _Cap()
        stop_all_requested = True
        stream_death_requested = False

    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(_HardExitCalled):
            loop.run_until_complete(_m._stop_capture_bounded(_App(), 0.2))
        assert cap.get("code") == 43, f"wedged capture.stop must hard-exit 43; got {cap}"
    finally:
        released.set()  # let the worker finish so the executor join is instant
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()


def test_stop_capture_bounded_continues_on_quick_capture_stop_error(monkeypatch):
    """A capture.stop that raises QUICKLY (not a wedge) means its worker already
    finished -- there is no abandoned executor thread to join, so teardown must
    log-and-continue, NOT hard-exit. Only a still-PENDING task (a real wedge) is
    dangerous; a task that is DONE (success or exception) is safe.

    Seeded defect: hard-exit whenever task.exception() is not None -> a benign
    capture.stop error wrongly hard-exits -> _HardExitCalled raised -> red.
    """
    import asyncio
    import yamabiko.main as _m

    def _fake_hard_exit(code):
        raise _HardExitCalled(code)
    monkeypatch.setattr(_m, "_hard_exit", _fake_hard_exit)

    class _Cap:
        def stop(self):
            raise RuntimeError("quick teardown error, worker already done")

    class _App:
        capture = _Cap()
        stop_all_requested = True
        stream_death_requested = False

    # Returns normally (no hard-exit) within the budget.
    asyncio.run(_m._stop_capture_bounded(_App(), 0.5))


def test_stop_capture_bounded_continues_on_quick_timeout_error(monkeypatch):
    """A capture.stop that raises TimeoutError QUICKLY must be treated as a
    benign finished-with-error (log + continue), NOT as a wedge -- the worker
    already completed, so there is no abandoned executor thread to join.

    This is the discrimination the wait-based implementation buys over wait_for:
    on Python 3.11+ asyncio.TimeoutError IS builtins.TimeoutError, and wait_for
    re-raises a worker exception, so the old `except asyncio.TimeoutError` would
    mistake this quick TimeoutError for a genuine budget timeout and hard-exit.

    Seeded defect: revert _stop_capture_bounded to
    `await asyncio.wait_for(...); except asyncio.TimeoutError: _hard_exit(...)`
    -> this quick TimeoutError is caught as a timeout -> _HardExitCalled -> red.
    """
    import asyncio
    import yamabiko.main as _m

    def _fake_hard_exit(code):
        raise _HardExitCalled(code)
    monkeypatch.setattr(_m, "_hard_exit", _fake_hard_exit)

    class _Cap:
        def stop(self):
            raise TimeoutError("device query timed out, but the call returned")

    class _App:
        capture = _Cap()
        stop_all_requested = True
        stream_death_requested = False

    # Must return normally (no hard-exit): the task is DONE, not pending.
    asyncio.run(_m._stop_capture_bounded(_App(), 0.5))


# --- teardown glossary save: issue-#55 class (P2, 2026-07-12 review) --------
# The old shape -- wait_for(shield(to_thread(save))) -- abandoned a wedged
# save worker on asyncio's DEFAULT executor; asyncio.run's Runner.close()
# then re-joins that executor (shutdown_default_executor, THREAD_JOIN_TIMEOUT
# == 300 s) BEFORE main() can reach its _hard_exit, so a wedged fsync kept
# yamabiko.exe alive minutes after tray-Quit -- the exact stall class
# _stop_capture_bounded hard-exits around. _save_glossary_bounded instead
# runs save() on a DEDICATED daemon thread and bounded-joins it: the join
# always returns (so the executor stays clean) and the daemon flag lets the
# process exit with the save still stuck.


def test_glossary_save_wedge_does_not_stall_asyncio_run_close(caplog):
    """A save() that never returns must not stall asyncio.run past the
    budget -- INCLUDING Runner.close()'s default-executor re-join, which is
    inside the measured window here.

    Seeded defect: revert _save_glossary_bounded to
    wait_for(shield(to_thread(save))) -> the coroutine still returns at the
    0.3 s budget, but Runner.close() joins the default-executor worker
    stuck in save() -> asyncio.run takes ~30 s (the fake's own release
    cap) -> red. The daemon assert is the second observable: only a
    daemon thread lets the real process exit with the save still wedged.
    """
    import logging
    import threading
    import time as _time
    import yamabiko.main as _m

    release = threading.Event()
    entered = threading.Event()

    class _Mgr:
        def save(self):
            entered.set()
            release.wait(timeout=30)  # the wedge (stuck fsync)

    caplog.set_level(logging.WARNING, logger="yamabiko")
    t0 = _time.monotonic()
    asyncio.run(_m._save_glossary_bounded(_Mgr(), 0.3))
    elapsed = _time.monotonic() - t0

    wedged = [
        t for t in threading.enumerate()
        if t.name == "glossary-save" and t.is_alive()
    ]
    try:
        assert entered.is_set(), "save() was never invoked (harness drift)"
        assert elapsed < 5.0, (
            f"asyncio.run took {elapsed:.1f}s against a wedged glossary "
            f"save; the save worker is being re-joined at Runner.close() "
            f"(issue-#55 class)"
        )
        assert wedged, (
            "no live glossary-save thread found -- the wedge never engaged "
            "or the save ran on the default executor under another name"
        )
        assert all(t.daemon for t in wedged), (
            "the abandoned save thread is NON-daemon; interpreter exit "
            "would join it and yamabiko.exe would outlive tray-Quit"
        )
        assert any(
            "glossary save" in r.getMessage() for r in caplog.records
        ), "no WARNING recorded the abandoned save"
    finally:
        release.set()
        for t in wedged:
            t.join(timeout=5)


def test_glossary_save_quick_completion_runs_save_without_abandon_warning(caplog):
    """The normal path: a fast save completes inside the budget, no
    abandon warning. Seeded defect: skip starting the save thread ->
    saved stays False -> red."""
    import logging
    import yamabiko.main as _m

    saved = {"done": False}

    class _Mgr:
        def save(self):
            saved["done"] = True

    caplog.set_level(logging.WARNING, logger="yamabiko")
    asyncio.run(_m._save_glossary_bounded(_Mgr(), 5.0))
    assert saved["done"] is True
    assert not any(
        "glossary save still running" in r.getMessage()
        for r in caplog.records
    ), "a quick save must not log the abandon warning"


def test_glossary_save_error_is_logged_not_raised(caplog):
    """A save() exception is logged by the worker and must not abort the
    rest of teardown (the helper returns normally). Seeded defect: drop
    the try/except in the save worker -> the exception dies unlogged on
    the daemon thread -> no 'glossary save failed' record -> red."""
    import logging
    import yamabiko.main as _m

    class _Mgr:
        def save(self):
            raise OSError(28, "No space left on device")

    caplog.set_level(logging.ERROR, logger="yamabiko")
    asyncio.run(_m._save_glossary_bounded(_Mgr(), 5.0))
    assert any(
        "glossary save failed" in r.getMessage() for r in caplog.records
    ), "the save exception was swallowed without a log"


@pytest.mark.timeout(30)
def test_run_pipeline_teardown_calls_bounded_glossary_save(
    _pipeline_env, tmp_path, caplog,
):
    """Wiring check (charter 3.9): run_pipeline's finally must route the
    glossary save through the bounded daemon-thread helper. A wedged save
    on a live PipelineApp returns within the grace-derived budget and
    leaves the abandoned daemon thread behind.

    Seeded defect: call app.glossary_mgr.save() inline (or via a bare
    to_thread await) in the finally -> teardown blocks on the 20 s wedge
    past the asserted bound -> red.
    """
    import logging
    import threading
    import time as _time

    env = _pipeline_env
    _write_overrides(tmp_path, {
        "lifecycle": {"grace_s": {"pipeline": GRACE_S}},
    })

    release = threading.Event()

    class _WedgedMgr:
        def save(self):
            release.wait(timeout=20)

    caplog.set_level(logging.WARNING, logger="yamabiko")

    async def _drive():
        args = env.module.parse_args([])
        app = await env.module.build_pipeline(args)
        app.glossary_mgr = _WedgedMgr()
        app.done.set()
        await env.module.run_pipeline(app)

    t0 = _time.monotonic()
    asyncio.run(_drive())
    elapsed = _time.monotonic() - t0
    release.set()

    assert elapsed < GRACE_S + _GRACE_ASSERT_SLACK_S, (
        f"run_pipeline teardown took {elapsed:.1f}s against a wedged "
        f"glossary save; the bounded daemon-thread save is not wired in"
    )
    assert any(
        "glossary save still running" in r.getMessage()
        for r in caplog.records
    ), "the teardown never hit the bounded-save abandon path (harness drift)"


def test_main_hard_exits_0_on_clean_graceful_stop(monkeypatch):
    """A clean graceful stop (named-event shutdown: amain returns 0,
    stop_all_requested False) must STILL hard-exit via os._exit(0) -- the
    named-event path is a real production path (SIGINT/SIGTERM, tray-Quit
    cascade, frozen-E2E graceful-shutdown test), not "dev-only", and a wedged
    capture.stop on it has no os._exit backstop without this (issue #61).
    os._exit(0) is identical to a normal return for the OS and the watchdog
    (classify(0) == STOP_SELF).

    Seeded defect: restore `if code: _hard_exit(code)` -> code 0 skips the
    hard-exit -> 'hard_exit_code' absent -> red.
    """
    import yamabiko.main as _m
    cap = _patch_main_preamble(monkeypatch, _m)

    class _App:
        stop_all_requested = False
        stream_death_requested = False

    async def _fake_amain(args, app_box=None):
        app_box["app"] = _App()
        return 0
    monkeypatch.setattr(_m, "amain", _fake_amain)

    with pytest.raises(_HardExitCalled):
        _m.main()
    assert cap.get("hard_exit_code") == 0, (
        f"clean graceful stop must os._exit(0); got {cap}"
    )


def test_main_hard_exits_43_on_startup_failure_systemexit(monkeypatch):
    """A startup failure (build_pipeline / _start_capture_or_exit show a tailored
    dialog then sys.exit(43)) propagates SystemExit(43) into main() -- and on a
    build_pipeline failure app_box is EMPTY (the app was never returned). main()
    must os._exit(43): re-raising runs interpreter finalization, which joins the
    never-shut-down cuda_pool CTranslate2/onnxruntime threads and can mangle the
    43 so the watchdog restart-loops the tailored dialog instead of stopping
    (issue #56).

    Seeded defect: drop the SystemExit(43) backstop in main()'s except handler
    -> SystemExit(43) re-raises, os._exit never called -> 'hard_exit_code'
    absent -> red.
    """
    import yamabiko.main as _m
    cap = _patch_main_preamble(monkeypatch, _m)

    async def _failing_amain(args, app_box=None):
        # app_box is NEVER populated: build_pipeline raised before returning.
        raise SystemExit(43)
    monkeypatch.setattr(_m, "amain", _failing_amain)

    with pytest.raises(_HardExitCalled):
        _m.main()
    assert cap.get("hard_exit_code") == 43, (
        f"startup failure (empty app_box) must os._exit(43); got {cap}"
    )


def test_main_does_not_coerce_non_43_systemexit_to_hard_exit(monkeypatch):
    """The startup-failure backstop must be SPECIFIC to code 43 (the tailored-
    dialog sentinel). A SystemExit with any other code is a genuine abnormal
    exit and must propagate, not be silently converted into a STOP_ALL hard-exit.

    Seeded defect: widen the backstop to `isinstance(exc, SystemExit)` (any
    code) -> SystemExit(1) becomes a hard-exit -> RuntimeError-style propagation
    is lost -> red.
    """
    import yamabiko.main as _m
    cap = _patch_main_preamble(monkeypatch, _m)

    async def _failing_amain(args, app_box=None):
        raise SystemExit(1)  # not the 43 startup-dialog sentinel
    monkeypatch.setattr(_m, "amain", _failing_amain)

    with pytest.raises(SystemExit) as ei:
        _m.main()
    assert ei.value.code == 1
    assert "hard_exit_code" not in cap, "a non-43 SystemExit must NOT hard-exit"


def test_run_pipeline_wires_browser_server_when_enabled(_pipeline_env, tmp_path):
    """When cfg['overlay']['browser'] is True, run_pipeline must:
      - call build_app(state, on_quit=app.request_stop_all)
      - stash the returned fastapi_app.state.manager on app.manager
      - construct uvicorn.Config with the configured port
      - pin the asyncio / h11 / websockets / lifespan-on backends
      - construct uvicorn.Server and stash it on app.uv_server
      - overwrite uv_server.install_signal_handlers with a no-op
        (so uvicorn doesn't steal SIGINT from the pipeline's
        KeyboardInterrupt path)

    The whole overlay front-end depends on this wiring: browser clients
    connect to :websocket_port and receive subtitle events via the
    manager. A regression skipping any of these steps means the
    overlay silently shows nothing while the pipeline runs normally.

    Seeded-defect checks: comment out `app.manager =
    fastapi_app.state.manager` -> the manager identity assertion fails. Delete
    any concrete Uvicorn selector -> the fake exposes Uvicorn's ``"auto"``
    default -> its corresponding backend assertion fails.
    """
    env = _pipeline_env
    _write_overrides(tmp_path, {
        "overlay": {"browser": True, "websocket_port": 9999},
    })

    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    # build_app called exactly once
    assert len(env.build_app_calls) == 1, (
        f"expected one build_app call; got {len(env.build_app_calls)}"
    )
    call = env.build_app_calls[0]
    assert call["state"] is app.app_state, (
        "build_app was called with a different AppState than the one "
        "on PipelineApp; state broadcasts will target the wrong object"
    )
    # Bound-method equality (not identity): on_quit is app.request_stop_all
    # captured at build_app call time. Python compares __self__ and __func__,
    # so a fresh `app.request_stop_all` still == the captured one. (The quit
    # callback fires `done` AND marks stop_all_requested -> amain exits
    # STOP_ALL 43; the flag-to-exit-code behaviour is covered by
    # test_amain_returns_43_on_user_quit.)
    assert call["on_quit"] == app.request_stop_all, (
        "build_app on_quit is not app.request_stop_all; the browser/tray Quit "
        "button won't trigger a STOP_ALL and the watchdog will restart the "
        "user's quit as a crash"
    )

    # The tray's set_hotkeys command must re-bind the pipeline's shortcuts.
    # Bound-method equality as for on_quit. Seeded defect: drop
    # on_hotkeys_change from the build_app call -> the fake records None -> red.
    assert call["on_hotkeys_change"] == app.apply_hotkeys, (
        "build_app on_hotkeys_change is not app.apply_hotkeys; shortcuts "
        "changed in Settings would not re-bind until a restart"
    )

    # Manager wired from fastapi_app.state.manager
    assert app.manager is env.fake_fastapi_app.state.manager, (
        "app.manager is not fastapi_app.state.manager; browser "
        "clients won't receive subtitle broadcasts"
    )

    # Uvicorn server constructed once, stored on app
    assert len(env.uvicorn_server_instances) == 1, (
        f"expected one uvicorn.Server construction; got "
        f"{len(env.uvicorn_server_instances)}"
    )
    assert app.uv_server is env.uvicorn_server_instances[0], (
        "app.uv_server is not the uvicorn.Server instance constructed "
        "inside run_pipeline"
    )
    assert app.uv_server.config.port == 9999, (
        f"uvicorn port did not propagate from cfg websocket_port; "
        f"got {app.uv_server.config.port}"
    )
    assert app.uv_server.config.host == "127.0.0.1", (
        f"uvicorn host not 127.0.0.1; got {app.uv_server.config.host}"
    )
    assert app.uv_server.config.loop == "asyncio", (
        f"uvicorn loop backend is not pinned to asyncio; "
        f"got {app.uv_server.config.loop!r}"
    )
    assert app.uv_server.config.http == "h11", (
        f"uvicorn HTTP backend is not pinned to h11; "
        f"got {app.uv_server.config.http!r}"
    )
    assert app.uv_server.config.ws == "websockets", (
        f"uvicorn WebSocket backend is not pinned to websockets; "
        f"got {app.uv_server.config.ws!r}"
    )
    assert app.uv_server.config.lifespan == "on", (
        f"uvicorn lifespan backend is not pinned on; "
        f"got {app.uv_server.config.lifespan!r}"
    )

    # install_signal_handlers overwritten (fake sentinel was a string,
    # code overwrites it with lambda: None).
    assert callable(app.uv_server.install_signal_handlers), (
        "install_signal_handlers not overwritten with no-op lambda; "
        "uvicorn will install its own SIGINT handler and steal "
        "the shutdown signal from the pipeline"
    )
    assert app.uv_server.install_signal_handlers() is None, (
        "install_signal_handlers overwrite is not the lambda: None "
        "no-op; could still interfere with shutdown"
    )

    # browser=True must request the browser page routes.
    assert env.build_app_calls[0]["serve_browser_ui"] is True, (
        "overlay.browser=true did not request the browser page routes"
    )

    # The configured websocket_port must reach build_app so the /ws Origin
    # guard pins browser origins to EXACTLY this server's port (build_app
    # requires bound_port). If run_pipeline passed the wrong port, the guard
    # would trust a co-resident localhost service's page on that other port.
    # Seeded defect: drop `bound_port=port` from the build_app(...)
    # call in run_pipeline -> the fake records its None default -> red.
    assert env.build_app_calls[0]["bound_port"] == 9999, (
        f"build_app got bound_port={env.build_app_calls[0]['bound_port']!r}; "
        f"expected the configured websocket_port 9999. The /ws Origin "
        f"guard is running port-blind."
    )

    # The tray's pause command arrives through on_pause_change. Invoke the
    # recorded callable, not just its identity: a callable that only flips
    # the level would leave capture and the unfinished-text owner on the
    # old epoch. Seeded defect: wire on_pause_change to a callable that only
    # calls app.app_state.set_paused -> the epoch and pause-edge assertions
    # go red.
    epoch_before = app.remainder_state.active_epoch
    asyncio.run(env.build_app_calls[0]["on_pause_change"](True))
    assert app.app_state.paused is True, (
        "on_pause_change(True) did not pause the shared AppState"
    )
    assert app.remainder_state.active_epoch > epoch_before, (
        "on_pause_change(True) left unfinished Japanese on the old epoch"
    )
    assert app.capture.pause_edges == [app.app_state.pause_snapshot()], (
        "on_pause_change(True) never ordered a pause edge into capture"
    )


def test_run_pipeline_passes_gazetteer_enabled_false_to_build_app_when_feature_off(
    _pipeline_env, tmp_path,
):
    """run_pipeline must thread feature_enabled(cfg) into build_app as
    gazetteer_enabled, so the /ws set_series handler rejects the command when
    the series feature is off. With gazetteer.enabled=False, the value passed
    must be False -- otherwise a token-free loopback client could drive the
    disabled feature over the wire.

    Asserting the FALSE case is what makes this seeded-defect-sensitive: the
    test double defaults gazetteer_enabled=True, so dropping the production
    `gazetteer_enabled=feature_enabled(app.cfg)` kwarg from the build_app call
    leaves the recorded value True -> this assertion goes red.
    """
    env = _pipeline_env
    _write_overrides(tmp_path, {
        "overlay": {"browser": True, "websocket_port": 9999},
        "gazetteer": {"enabled": False},
        "romanization": {"enabled": True},
    })

    asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert len(env.build_app_calls) == 1
    assert env.build_app_calls[0]["gazetteer_enabled"] is False, (
        "build_app was not told the gazetteer feature is off; the /ws "
        "set_series handler would accept a disabled-feature command from any "
        "loopback client"
    )


def test_run_pipeline_starts_ws_server_even_with_browser_ui_disabled(
    _pipeline_env, tmp_path, monkeypatch,
):
    """The uvicorn/FastAPI server is the display + control bus —
    /ws feeds the Qt overlay and tray commands, /healthz + /livez feed
    the watchdog probes. overlay.browser=false (the natural override
    to drop the browser tab) used to skip the WHOLE server: subtitles
    appeared nowhere, tray Quit became a no-op leaving pipeline +
    llama running, and the watchdog probe never armed. The server now
    starts unconditionally; the key only gates the page routes.

    Seeded defect: restore the `if app.cfg["overlay"].get("browser", ...):`
    guard around the server construction -> build_app_calls stays
    empty with browser=False -> red.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)  # fixture default has browser=False

    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert len(env.build_app_calls) == 1, (
        f"build_app must be called exactly once regardless of "
        f"overlay.browser; got {len(env.build_app_calls)}"
    )
    assert env.build_app_calls[0]["serve_browser_ui"] is False, (
        "overlay.browser=false must gate the browser page routes "
        "(serve_browser_ui=False), not the server"
    )
    assert len(env.uvicorn_server_instances) == 1, (
        f"uvicorn.Server must be constructed regardless of "
        f"overlay.browser; got {len(env.uvicorn_server_instances)}"
    )
    assert app.manager is env.fake_fastapi_app.state.manager, (
        "app.manager not wired with browser=false; the Qt overlay "
        "would never receive subtitle broadcasts"
    )
    assert app.uv_server is env.uvicorn_server_instances[0], (
        "app.uv_server not wired with browser=false; /healthz and "
        "/livez would never come up for the watchdog probes"
    )


def test_overlay_ws_url_honors_configured_websocket_port(_pipeline_env):
    """Helper contract: overlay.websocket_port flows into the
    ws:// URL handed to the spawned overlay; absent key falls back to
    the same 8765 default uvicorn uses.

    Seeded defect: hardcode 8765 in _overlay_ws_url -> the 9999
    assertion goes red.
    """
    env = _pipeline_env
    assert env.module._overlay_ws_url(
        {"overlay": {"websocket_port": 9999}}
    ) == "ws://127.0.0.1:9999/ws"
    assert env.module._overlay_ws_url({}) == "ws://127.0.0.1:8765/ws"
    assert env.module._overlay_ws_url(
        {"overlay": {"websocket_port": "9001"}}
    ) == "ws://127.0.0.1:9001/ws", "YAML string port must coerce like uvicorn's int() does"


def test_run_pipeline_spawns_overlay_with_configured_ws_url(
    _pipeline_env, tmp_path, monkeypatch,
):
    """uvicorn and the watchdog probe both honor overlay.websocket_port,
    but the spawned Qt overlay child kept the hardcoded
    ws://127.0.0.1:8765/ws argparse default — a user who moves the port
    off 8765 got a silent no-subtitle overlay stuck in reconnect backoff,
    with tray Pause/Genre/Quit as silent no-ops.

    The test captures the spawn closure run_pipeline wires into the
    overlay PipelineSupervisor, INVOKES it (wired callables must be
    exercised, not just registered), and asserts the observable effect:
    the Popen argv carries the configured URL.

    Seeded defect: drop the `url=_overlay_ws_url(app.cfg)` argument
    from the _spawn_overlay -> _build_overlay_cmd call -> the argv
    has no --url -> red.
    """
    import subprocess as _sp

    import yamabiko.lifecycle.pipeline_supervisor as _ps_mod

    env = _pipeline_env
    _write_overrides(tmp_path, {"overlay": {"websocket_port": 9999}})

    captured: dict = {}
    _RealSupervisor = _ps_mod.PipelineSupervisor

    class _CapturingSupervisor(_RealSupervisor):
        def __init__(self, *a, **kw):
            captured["spawn"] = kw.get("spawn")
            super().__init__(*a, **kw)

        def run(self):
            # Don't supervise: the test invokes the spawn closure
            # itself, deterministically, after run_pipeline returns.
            return

    monkeypatch.setattr(_ps_mod, "PipelineSupervisor", _CapturingSupervisor)

    asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert callable(captured.get("spawn")), (
        "run_pipeline did not wire a spawn closure into the overlay "
        "supervisor"
    )

    spawn_cmds: list[list[str]] = []

    class _RecordingProc:
        pid = 999999
        returncode = None

        def poll(self):
            return self.returncode

    spawn_envs: list[dict] = []

    def _recording_popen(cmd, *a, **kw):
        spawn_cmds.append(list(cmd))
        spawn_envs.append(kw.get("env") or {})
        return _RecordingProc()

    monkeypatch.setattr(_sp, "Popen", _recording_popen)
    captured["spawn"]()

    assert spawn_cmds, "spawn closure did not Popen an overlay child"
    argv = spawn_cmds[-1]
    assert "--url" in argv, f"overlay spawn argv lacks --url: {argv}"
    assert argv[argv.index("--url") + 1] == "ws://127.0.0.1:9999/ws", (
        f"overlay spawn argv carries the wrong WS URL: {argv}"
    )
    # The server refuses token-free clients, so the overlay child must get
    # the SAME session token run_pipeline handed to build_app.
    # Seeded defect: drop `env[WS_TOKEN_ENV] = ws_token` from
    # _overlay_cmd_env -> the env lacks the token -> red.
    from yamabiko.display.broadcast import WS_TOKEN_ENV
    server_token = env.build_app_calls[0]["ws_token"]
    assert server_token, "run_pipeline built the server without a ws_token"
    assert spawn_envs[-1].get(WS_TOKEN_ENV) == server_token, (
        "the overlay child's environment does not carry the server's "
        "session token; its WebSocket would be refused"
    )


def _capture_overlay_command_from_run_pipeline(env, args, monkeypatch):
    import subprocess as _sp

    import yamabiko.lifecycle.pipeline_supervisor as _ps_mod

    captured: dict = {}
    real_supervisor = _ps_mod.PipelineSupervisor

    class _CapturingSupervisor(real_supervisor):
        def __init__(self, *a, **kw):
            captured["spawn"] = kw.get("spawn")
            super().__init__(*a, **kw)

        def run(self):
            return

    monkeypatch.setattr(_ps_mod, "PipelineSupervisor", _CapturingSupervisor)

    async def _drive() -> None:
        app = await env.module.build_pipeline(args)
        app.done.set()
        await env.module.run_pipeline(app)

    asyncio.run(_drive())
    assert callable(captured.get("spawn"))

    spawn_cmds: list[list[str]] = []

    class _RecordingProc:
        pid = 999999
        returncode = None

        def poll(self):
            return self.returncode

    def _recording_popen(cmd, *a, **kw):
        spawn_cmds.append(list(cmd))
        return _RecordingProc()

    monkeypatch.setattr(_sp, "Popen", _recording_popen)
    captured["spawn"]()
    assert spawn_cmds, "spawn closure did not Popen an overlay child"
    return spawn_cmds[-1]


def test_run_pipeline_explicit_config_reaches_overlay_child_argv(
    _pipeline_env, tmp_path, monkeypatch,
):
    config_path = tmp_path / "repro.yaml"
    config_path.write_text(
        "logging:\n  level: DEBUG\n",
        encoding="utf-8",
    )
    args = _pipeline_env.module.parse_args(["--config", str(config_path)])

    cmd = _capture_overlay_command_from_run_pipeline(
        _pipeline_env, args, monkeypatch,
    )

    assert ["--config", str(config_path)] == cmd[
        cmd.index("--config"):cmd.index("--config") + 2
    ]


def test_run_pipeline_overlay_log_level_round_trips_from_explicit_config(
    _pipeline_env, tmp_path, monkeypatch,
):
    """``logging.level`` must reach the spawned overlay's argv FROM THE
    CONFIG THE PIPELINE ACTUALLY RUNS WITH (app.cfg) -- here an explicit
    ``--config`` file carrying DEBUG. _build_overlay_cmd used to do a
    fresh ``load_full_config()`` disk read from inside the pipeline
    process, which silently ignores ``--config`` (the documented bug
    class from run_pipeline's lifecycle block): with LOCALAPPDATA
    pointed at a dir with NO user_overrides.yaml, that fresh read
    yields the shipped INFO default, so this round-trip can only pass
    when the level flows app.cfg -> spawn closure -> argv.
    Deliberately NO load_full_config monkeypatching -- that mocked seam
    is exactly the gap this test exists to close (§3.2).

    Seeded-defect drill: drop the ``level=...`` argument from
    _overlay_cmd_env's _build_overlay_cmd call (or re-add the fresh
    load_full_config() read inside _build_overlay_cmd) -> the argv
    carries --log-level INFO, not DEBUG -> red.
    """
    import subprocess as _sp

    import yaml as _yaml

    import yamabiko.lifecycle.pipeline_supervisor as _ps_mod
    from yamabiko import paths as _paths

    env = _pipeline_env
    # Full --config file built from the shipped schema (the explicit
    # file is layered over shipped defaults, user_overrides.yaml
    # skipped), heavy collaborators disabled to match the fixture's
    # stubs, logging.level: DEBUG as the value that must round-trip.
    # NOTE: no _write_overrides here --
    # the layered config store must NOT contain DEBUG, or a fresh
    # load_full_config() inside the spawn path would see it and this
    # test could not distinguish app.cfg from a disk re-read.
    cfg = _yaml.safe_load(
        _paths.shipped_defaults_path().read_text(encoding="utf-8")
    )
    cfg["romanization"]["enabled"] = False
    cfg["hotkeys"]["enabled"] = False
    cfg["overlay"]["browser"] = False
    cfg["logging"]["level"] = "DEBUG"
    explicit = tmp_path / "explicit-debug.yaml"
    explicit.write_text(_yaml.safe_dump(cfg), encoding="utf-8")

    captured: dict = {}
    _RealSupervisor = _ps_mod.PipelineSupervisor

    class _CapturingSupervisor(_RealSupervisor):
        def __init__(self, *a, **kw):
            captured["spawn"] = kw.get("spawn")
            super().__init__(*a, **kw)

        def run(self):
            # Don't supervise: the test invokes the spawn closure
            # itself, deterministically, after run_pipeline returns.
            return

    monkeypatch.setattr(_ps_mod, "PipelineSupervisor", _CapturingSupervisor)

    async def _drive() -> None:
        args = env.module.parse_args(["--config", str(explicit)])
        app = await env.module.build_pipeline(args)
        app.done.set()
        await env.module.run_pipeline(app)

    asyncio.run(_drive())

    assert callable(captured.get("spawn")), (
        "run_pipeline did not wire a spawn closure into the overlay "
        "supervisor"
    )

    spawn_cmds: list[list[str]] = []

    class _RecordingProc:
        pid = 999999
        returncode = None

        def poll(self):
            return self.returncode

    def _recording_popen(cmd, *a, **kw):
        spawn_cmds.append(list(cmd))
        return _RecordingProc()

    monkeypatch.setattr(_sp, "Popen", _recording_popen)
    captured["spawn"]()

    assert spawn_cmds, "spawn closure did not Popen an overlay child"
    argv = spawn_cmds[-1]
    assert "--log-level" in argv, (
        f"overlay spawn argv lacks --log-level: {argv}"
    )
    assert argv[argv.index("--log-level") + 1] == "DEBUG", (
        f"overlay spawn argv carries --log-level "
        f"{argv[argv.index('--log-level') + 1]!r}, not the explicit "
        f"--config file's DEBUG: the spawn path is reading config from "
        f"disk instead of app.cfg (the fresh-load_full_config bug class)"
    )


def test_run_pipeline_wires_atomic_spawn_in_job_for_overlay(
    _pipeline_env, tmp_path, monkeypatch,
):
    """run_pipeline must hand PipelineSupervisor a ``spawn_in_job``
    callable that routes through jobs.spawn_in_job, so the overlay is
    assigned to the inner job while still CREATE_SUSPENDED. The legacy
    two-step path (Popen, then assign) leaves a window where a
    force-killed pipeline strands a running overlay OUTSIDE the job --
    it survives KILL_ON_JOB_CLOSE and the watchdog's pipeline respawn
    then puts a second overlay on screen.

    The test captures the callable run_pipeline wires in, INVOKES it,
    and asserts the jobs.spawn_in_job seam receives:
    the configured --url argv (parity with the legacy closure), the
    exact job object handed to the callable, the overlay shutdown-event
    env var, and DEVNULL stdin.

    Seeded defect: drop the ``spawn_in_job=_spawn_overlay_in_job`` kwarg
    from the PipelineSupervisor(...) call in run_pipeline -> captured
    value is None -> red.
    """
    import subprocess as _sp

    import yamabiko.lifecycle.jobs as _jobs_mod
    import yamabiko.lifecycle.pipeline_supervisor as _ps_mod
    from yamabiko.lifecycle.shutdown_event import ENV_VAR_BY_ROLE

    env = _pipeline_env
    _write_overrides(tmp_path, {"overlay": {"websocket_port": 9999}})

    captured: dict = {}
    _RealSupervisor = _ps_mod.PipelineSupervisor

    class _CapturingSupervisor(_RealSupervisor):
        def __init__(self, *a, **kw):
            captured["spawn_in_job"] = kw.get("spawn_in_job")
            super().__init__(*a, **kw)

        def run(self):
            # Don't supervise: the test invokes the captured callable
            # itself, deterministically, after run_pipeline returns.
            return

    monkeypatch.setattr(_ps_mod, "PipelineSupervisor", _CapturingSupervisor)

    seam_calls: list[dict] = []

    class _RecordingProc:
        pid = 999998
        returncode = None

        def poll(self):
            return self.returncode

    def _recording_spawn_in_job(cmd, job, *, env=None, **kw):
        seam_calls.append({"cmd": list(cmd), "job": job, "env": env, "kw": kw})
        return _RecordingProc()

    # Re-patch over _stub_overlay_runtime's generic stub so this test
    # records the full call shape (last patch wins).
    monkeypatch.setattr(_jobs_mod, "spawn_in_job", _recording_spawn_in_job)

    asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert callable(captured.get("spawn_in_job")), (
        f"run_pipeline wired spawn_in_job={captured.get('spawn_in_job')!r} "
        f"into the overlay supervisor; production is still on the legacy "
        f"racy two-step spawn path"
    )

    sentinel_job = object()
    proc = captured["spawn_in_job"](sentinel_job)
    assert isinstance(proc, _RecordingProc), (
        "the wired spawn_in_job callable did not return the seam's proc"
    )
    assert seam_calls, (
        "invoking the wired callable never reached jobs.spawn_in_job"
    )
    call = seam_calls[-1]
    assert call["job"] is sentinel_job, (
        "the callable did not forward the supervisor's job object to "
        "jobs.spawn_in_job; the overlay would join the wrong (or no) job"
    )
    argv = call["cmd"]
    assert "--url" in argv and argv[argv.index("--url") + 1] == (
        "ws://127.0.0.1:9999/ws"
    ), (
        f"atomic spawn argv lost the configured WS URL: {argv}"
    )
    assert ENV_VAR_BY_ROLE["overlay"] in (call["env"] or {}), (
        "the overlay shutdown-event env var is missing from the atomic "
        "spawn env; the overlay child can't hear the graceful shutdown "
        "ping"
    )
    assert call["kw"].get("stdin") is _sp.DEVNULL, (
        f"atomic spawn lost stdin=DEVNULL (got "
        f"{call['kw'].get('stdin')!r}); the overlay child inherits the "
        f"pipeline's stdin"
    )


def test_run_pipeline_server_drain_wait_exceeds_uvicorn_drain_bound(
    _pipeline_env, tmp_path, monkeypatch,
):
    """Two timeouts govern the server teardown -- uvicorn's own
    drain bound (timeout_graceful_shutdown, given to uvicorn.Config)
    and the outer asyncio.wait_for that lets uvicorn finish that drain
    in run_pipeline's finally. Outer <= inner cancels the server task
    MID-DRAIN whenever a WS client is slow to close (the shipped 3 s
    outer vs 5 s drain mismatch): dropped close frames + 'Task was
    destroyed while pending' -- exactly what the drain bound exists to
    prevent. Both values must come from the shared module constants and
    the outer must strictly exceed the inner, with the sum inside the
    watchdog's 10 s pipeline grace.

    The test records every asyncio.wait_for timeout used while
    run_pipeline tears down and asserts the shared outer constant was
    actually used AND exceeds the drain bound uvicorn received.

    Seeded defects (each one line):
      - revert the outer wait to ``timeout=3.0`` -> no recorded timeout
        matches _SERVER_DRAIN_WAIT_S -> red.
      - revert uvicorn.Config to ``timeout_graceful_shutdown=5`` while
        setting _SERVER_DRAIN_WAIT_S below it -> relationship assert red.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)

    recorded: list = []
    real_wait_for = asyncio.wait_for

    async def _recording_wait_for(awaitable, timeout=None, **kw):
        recorded.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout, **kw)

    monkeypatch.setattr(asyncio, "wait_for", _recording_wait_for)

    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    drain = app.uv_server.config.extra_kwargs.get("timeout_graceful_shutdown")
    assert drain is not None, (
        "uvicorn.Config no longer bounds its graceful drain "
        "(timeout_graceful_shutdown missing); spec C3 regression"
    )
    outer = getattr(env.module, "_SERVER_DRAIN_WAIT_S", None)
    assert outer is not None, (
        "main.py no longer defines _SERVER_DRAIN_WAIT_S; the outer "
        "server-drain wait and uvicorn's drain bound have been decoupled "
        "(server-drain regression class)"
    )
    assert outer in recorded, (
        f"run_pipeline's teardown never awaited with the shared "
        f"_SERVER_DRAIN_WAIT_S={outer}; recorded wait_for timeouts: "
        f"{recorded!r}. The outer wait has drifted from the constant."
    )
    assert outer > drain, (
        f"outer server-drain wait ({outer}s) does not exceed uvicorn's "
        f"own drain bound ({drain}s); the server task gets cancelled "
        f"mid-drain whenever a WS client is slow to close"
    )
    # Both must fit inside the watchdog's pipeline grace window or the
    # watchdog terminates the pipeline mid-teardown.
    assert outer < 10.0, (
        f"outer server-drain wait ({outer}s) exceeds the watchdog's "
        f"10 s lifecycle.grace_s.pipeline window"
    )


def test_run_pipeline_joins_overlay_supervisor_before_closing_its_handles(
    _pipeline_env, tmp_path, monkeypatch,
):
    """run_pipeline's finally must JOIN the overlay-supervisor
    thread (bounded by grace + headroom) before closing overlay_writer
    and inner_job. Without the join, the main thread races the
    supervisor's lock-release -> writer.signal() window: it closes the
    writer first, the supervisor's graceful SetEvent ping silently
    no-ops against handle=None, and the overlay falls through to the
    terminate/kill ladder on every shutdown.

    Deterministic schedule instead of a timing lottery: a stub
    supervisor thread that (like the real _shutdown_proc) signals the
    writer only AFTER stop is requested -- 0.2 s later -- and a
    recording writer that pins the signal-vs-close order.

    Seeded defect (one line): delete the ``overlay_thread.join(...)``
    from run_pipeline's finally -> close lands during the stub's 0.2 s
    window -> order == ['close', 'signal'] -> red.
    """
    import time as _time

    import yamabiko.lifecycle.pipeline_supervisor as _ps_mod
    import yamabiko.lifecycle.shutdown_event as _se_mod

    env = _pipeline_env
    _write_overrides(tmp_path)

    order: list[str] = []

    class _RecordingWriter:
        def __init__(self, name):
            self.name = name
            self.handle = object()  # truthy: close() has work to do

        def signal(self):
            order.append("signal")
            return self.handle is not None

        def close(self):
            order.append("close")
            self.handle = None

    monkeypatch.setattr(
        _se_mod.ShutdownEventWriter, "create",
        classmethod(lambda cls, role, parent_pid: _RecordingWriter(
            name=f"Local\\yamabiko.test.{role}.{parent_pid}",
        )),
    )

    class _LateSignalSupervisor:
        """Models the supervisor thread winning the stop race: it owns
        the shutdown ladder and delivers the graceful writer ping a
        beat after the stop request -- the exact window the missing
        join exposed."""

        def __init__(self, **kw):
            self._writer = kw["shutdown_writer"]
            self._stop_event = kw["stop_event"]

        def run(self):
            self._stop_event.wait(timeout=10.0)
            _time.sleep(0.2)
            self._writer.signal()

        def stop(self):
            # Main-thread stop() loses the race: the supervisor thread
            # already owns the ladder (idempotent-return path).
            return

    monkeypatch.setattr(
        _ps_mod, "PipelineSupervisor",
        lambda **kw: _LateSignalSupervisor(**kw),
    )

    asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert "signal" in order, (
        "the supervisor thread never delivered its writer ping; the "
        "test harness has drifted from run_pipeline's teardown"
    )
    assert "close" in order, (
        "run_pipeline's finally never closed overlay_writer; the Win32 "
        "event handle leaks"
    )
    assert order.index("signal") < order.index("close"), (
        f"overlay_writer was closed BEFORE the supervisor thread's "
        f"graceful signal (observed order {order}); without the "
        f"overlay_thread.join the SetEvent ping lands on a dead handle "
        f"and the overlay is always terminate/killed"
    )


@pytest.mark.timeout(25)
def test_run_pipeline_teardown_budget_bounds_wedged_supervisor_stop(
    _pipeline_env, tmp_path, monkeypatch, caplog,
):
    """A wedged overlay-supervisor stop() cannot push run_pipeline teardown
    past the grace-derived budget. The fix gives the finally-block ONE
    shared deadline (lifecycle.grace_s.pipeline - 2.0, floor 2.5 s),
    starts the stop() ladder BEFORE the server drain, bounds the drain by
    min(_SERVER_DRAIN_WAIT_S, remaining), and spends only the remaining
    budget on the ladder await and the overlay_thread.join. Without the
    budget, a stop() that blocks (hung overlay-process wait, dead Win32
    handle) holds teardown until the watchdog hard-kills the pipeline
    mid-cleanup -- losing glossary save / translator close / CUDA
    teardown.

    The stub supervisor wedges BOTH stop() and its run() thread on a
    30 s event that is only released after run_pipeline returns, so the
    whole measured window runs against a genuinely wedged ladder.
    grace=5.0 via overrides -> budget = max(5-2, 2.5) = 3.0 s; teardown
    should complete in ~3.5 s (budgeted ladder await + clamped 0.5 s
    join). The wall-clock assert allows grace + 3.0 s scheduler slack
    -- far below the 30 s wedge. The caplog assert proves the deadline
    path actually FIRED (TimeoutError branch -> ladder warning), so a
    green can't come from the ladder never wedging.

    Local @pytest.mark.timeout(25): the seeded-defect drill hangs until
    the 30 s wedge expires; the test-local bound reds it in reasonable
    time instead of the suite-default 180 s.

    Seeded defect: change the ladder await's ``timeout=_td_remaining()``
    to ``timeout=None`` in run_pipeline's finally -> teardown blocks on
    the wedged stop() past 25 s -> red.
    """
    import logging
    import threading
    import time as _time

    import yamabiko.lifecycle.pipeline_supervisor as _ps_mod

    env = _pipeline_env
    _write_overrides(tmp_path, {
        "lifecycle": {"grace_s": {"pipeline": GRACE_S}},
    })

    wedge_release = threading.Event()
    stop_calls: list[float] = []

    class _WedgedSupervisor:
        """Models the failure class the budget exists to bound:
        the shutdown ladder blocks indefinitely (e.g. a hung
        overlay-process wait inside _shutdown_proc)."""

        def __init__(self, **kw):
            self._stop_event = kw["stop_event"]

        def run(self):
            # Real supervisor loop: parks until the pipeline's
            # teardown sets the stop event... then wedges, so the
            # overlay_thread.join must ALSO be deadline-bounded.
            self._stop_event.wait(timeout=20.0)
            wedge_release.wait(30.0)

        def stop(self):
            stop_calls.append(_time.monotonic())
            wedge_release.wait(30.0)

    monkeypatch.setattr(
        _ps_mod, "PipelineSupervisor",
        lambda **kw: _WedgedSupervisor(**kw),
    )

    caplog.set_level(logging.WARNING, logger="yamabiko")

    async def _drive():
        args = env.module.parse_args([])
        app = await env.module.build_pipeline(args)
        app.done.set()
        await env.module.run_pipeline(app)
        # Release the wedge only AFTER run_pipeline returned: the
        # measured teardown window ran fully wedged, but asyncio.run's
        # own shutdown (which joins the to_thread executor worker
        # carrying stop()) must not serialise on the remaining ~26 s
        # of wedge and pollute the wall-clock measurement.
        wedge_release.set()

    t0 = _time.monotonic()
    asyncio.run(_drive())
    elapsed = _time.monotonic() - t0

    assert stop_calls, (
        "run_pipeline's teardown never started the supervisor stop() "
        "ladder; the wedge was not engaged and this test measured "
        "nothing (harness drift)"
    )
    assert elapsed < GRACE_S + _GRACE_ASSERT_SLACK_S, (
        f"run_pipeline teardown took {elapsed:.1f}s against a wedged "
        f"supervisor stop(); the grace-derived budget "
        f"(lifecycle.grace_s.pipeline={GRACE_S} -> {GRACE_S - 2.0} s) failed "
        f"to bound the ladder await / overlay_thread.join"
    )
    assert elapsed < 15.0, (
        f"run_pipeline teardown took {elapsed:.1f}s -- past any "
        f"plausible budget; the watchdog would hard-kill the pipeline "
        f"mid-cleanup"
    )
    ladder_warnings = [
        r for r in caplog.records
        if "overlay shutdown ladder still running at the teardown "
           "deadline" in r.getMessage()
    ]
    assert ladder_warnings, (
        f"the teardown-deadline TimeoutError path never fired -- the "
        f"test proved nothing about the budget (records: "
        f"{[r.getMessage() for r in caplog.records]!r})"
    )


@pytest.mark.timeout(25)
def test_run_pipeline_teardown_budget_bounds_cancel_swallowing_stage(
    _pipeline_env, tmp_path, monkeypatch, caplog,
):
    """The stage-cancel drain is charged to the SAME teardown budget as
    the server drain / overlay ladder / join. A stage that does not yield
    to cancellation (drain semantics swallowing CancelledError around
    in-flight work) must not push run_pipeline's finally past the
    grace-derived deadline.

    Production stages unwind on cancel (the plain ``await
    run_in_executor`` is interrupted; only the worker thread lingers in
    cuda_pool, which closes wait=False) -- so this pins the budget
    MECHANISM against the class with a synthetic swallowing stage.
    It also pins the asyncio.wait-not-wait_for choice: wait_for(gather)
    waits for the inner future's CANCELLATION to complete, which a
    swallowing child holds up indefinitely -- the exact trap the
    implementation avoids.

    grace=5.0 via overrides -> budget 3.0 s; the wedged stage ignores
    cancel behind a shield for up to 30 s. Wall-clock must stay within
    grace + slack, and the still-draining warning must fire (a green
    can't come from the stage never wedging).

    Seeded defect: revert the bounded
    ``asyncio.wait(pipeline_tasks, timeout=_td_remaining())`` to
    ``await asyncio.gather(*pipeline_tasks, return_exceptions=True)``
    -> teardown blocks on the swallowing stage past 25 s -> red via
    the test-local timeout.
    """
    import logging
    import threading
    import time as _time

    env = _pipeline_env
    _write_overrides(tmp_path, {
        "lifecycle": {"grace_s": {"pipeline": GRACE_S}},
    })

    wedge_release = threading.Event()
    entered = asyncio.Event()

    async def _wedged_translation_stage(*_a, **_kw):
        fut = asyncio.ensure_future(asyncio.to_thread(wedge_release.wait, 30.0))
        entered.set()
        while True:
            try:
                await asyncio.shield(fut)
                return
            except asyncio.CancelledError:
                # Drain semantics: finish the in-flight work; the shield
                # keeps fut alive across each swallowed cancel. Once the
                # work IS done (incl. asyncio.run's final wind-down
                # cancelling fut directly), yield -- a stage that
                # swallows cancel forever would deadlock the event
                # loop's own shutdown, which is not the modeled class.
                if fut.done():
                    return
                continue

    monkeypatch.setattr(
        env.module, "translation_stage", _wedged_translation_stage,
    )
    caplog.set_level(logging.WARNING, logger="yamabiko")

    async def _drive():
        args = env.module.parse_args([])
        app = await env.module.build_pipeline(args)
        app.done.set()
        await env.module.run_pipeline(app)
        # Release only after run_pipeline returned so the measured
        # teardown ran fully wedged; the still-pending stage task then
        # finishes promptly and asyncio.run's own wind-down stays fast.
        wedge_release.set()

    # httpx defers the httpcore transport import until the first client
    # is constructed; in this test that's run_pipeline's health prober
    # (the fixture replaces Translator, which normally pays the import
    # in build_pipeline at startup). On a cold process it's a
    # multi-second event-loop-blocking import that would be charged to
    # the measured teardown window. Pay it here, before t0.
    import httpx

    asyncio.run(httpx.AsyncHTTPTransport().aclose())

    t0 = _time.monotonic()
    asyncio.run(_drive())
    elapsed = _time.monotonic() - t0

    assert entered.is_set(), (
        "the wedged stage never started; run_pipeline did not create "
        "the translation stage task and this test measured nothing"
    )
    assert elapsed < GRACE_S + _GRACE_ASSERT_SLACK_S, (
        f"run_pipeline teardown took {elapsed:.1f}s against a "
        f"cancel-swallowing stage; the stage drain is not charged to "
        f"the grace-derived teardown budget"
    )
    drain_warnings = [
        r for r in caplog.records
        if "still draining at the teardown deadline" in r.getMessage()
    ]
    assert drain_warnings, (
        f"the bounded stage-drain deadline path never fired -- the "
        f"test proved nothing about the budget (records: "
        f"{[r.getMessage() for r in caplog.records]!r})"
    )


def test_run_pipeline_does_not_self_assign_to_inner_job(
    _pipeline_env, tmp_path, monkeypatch,
):
    """B1 regression guard: run_pipeline MUST NOT self-assign the current
    process to its inner Job Object.

    Background: the inner job is configured with KILL_ON_JOB_CLOSE.
    If pipeline self-assigns (i.e. adds its own PID to the job),
    then the `inner_job.close()` call in the finally block kills
    the pipeline itself via the kernel's kill-on-close cascade.
    In production this was masked by the pipeline already exiting;
    in tests the "pipeline PID" is the pytest runner PID, so
    pytest silently terminated with exit 0, swallowing every test
    after the first asyncio.run(run_pipeline(...)) in the suite.

    This test fails RED if anyone reintroduces the self-assign.
    It does NOT mock away the self-kill (that would just hide the
    bug again); instead it intercepts create_inner_job, hands back
    a tracking stub, and asserts the stub's assign() was never
    called with None (the sentinel that means "self").

    Seeded-defect drill: add back `inner_job.assign(None)` in
    run_pipeline. This assertion goes red with a clear message
    pointing at the self-assign. Also: if someone swaps the
    overlay assignment from overlay-Popen to None (mass-rename
    regression), this test likewise fails.

    This test intentionally does NOT assert that inner_job.assign
    was called with the overlay Popen -- overlay spawn happens on
    a background thread and races the early-done path. Coverage
    for the overlay-assign side lives in
    tests/lifecycle/test_pipeline_supervisor.py.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)

    # Tracking stub that stands in for the real JobObject but
    # records every assign() call. Mirrors the real close() no-op
    # shape (no handle, so no cascade).
    class _TrackingJob:
        def __init__(self, name: str):
            self.name = name
            self.assign_calls: list = []
            self.closed = False
            self.handle = None  # matches JobObject's "no-op" shape

        def assign(self, proc_or_pid) -> bool:
            self.assign_calls.append(proc_or_pid)
            return True

        def close(self) -> None:
            self.closed = True

    created_jobs: list[_TrackingJob] = []

    def _fake_create_inner_job(name: str) -> _TrackingJob:
        job = _TrackingJob(name)
        created_jobs.append(job)
        return job

    # Patch at the jobs module level: run_pipeline does
    # `from yamabiko.lifecycle.jobs import create_inner_job`
    # at call time, so patching the source module is what wires
    # through.
    import yamabiko.lifecycle.jobs as _jobs_mod
    monkeypatch.setattr(_jobs_mod, "create_inner_job", _fake_create_inner_job)

    asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert len(created_jobs) == 1, (
        f"expected exactly one inner job created by run_pipeline; "
        f"got {len(created_jobs)}"
    )
    job = created_jobs[0]

    # The core invariant: pipeline must NOT add itself (None) to
    # the inner job. Any assign(None) call is the B1 bug.
    self_assigns = [c for c in job.assign_calls if c is None]
    assert len(self_assigns) == 0, (
        "run_pipeline self-assigned to inner_job "
        f"({len(self_assigns)} time(s) with proc_or_pid=None). "
        "This is the B1 regression: inner_job has KILL_ON_JOB_CLOSE, "
        "so closing its handle in the finally block kills the "
        "pipeline PID itself. In tests this silently terminates "
        "pytest with exit 0. Nesting must form via overlay's explicit "
        "assign only; pipeline's outer-job inheritance + overlay's "
        "inner-job assign are sufficient to nest the jobs."
    )


# ---------------------------------------------------------------------------
# _start_capture_or_exit
#
# Extracted helper that wraps app.capture.start() so a device-resolution
# failure shows a tailored exit-43 dialog instead of propagating as a
# generic crash the watchdog restarts 3x (each reloading Whisper onto
# the GPU) before a generic give-up dialog.
#
# Tests are DIRECT (not wired-through-run_pipeline): the helper is
# module-level so it's directly callable and both tests below INVOKE it
# and assert on its side effects. The wiring test (run_pipeline calls
# _start_capture_or_exit) is exercised by the existing run_pipeline tests
# via the _FakeCapture fixture whose start() no-ops cleanly.
# ---------------------------------------------------------------------------


def test_start_capture_or_exit_shows_dialog_and_exits_43_on_device_failure(
    monkeypatch,
    caplog,
):
    """A capture.start() device-resolution failure must show a tailored
    device-gone dialog and sys.exit(43) (STOP_ALL) -- not propagate
    as a generic crash the watchdog restarts 3x while reloading Whisper
    onto the GPU each time.

    The real classify_startup_exception is NOT patched so the
    "Loopback device not found: '<name>'" form is also exercised
    end-to-end -- the title + device-name assertions prove both the
    guard AND the classifier work.

    Seeded-defect drill:
      - Remove the try/except in _start_capture_or_exit (let
        capture.start() run bare) -> RuntimeError propagates, no
        SystemExit(43) -> red.
      - Change sys.exit(43) to sys.exit(0) -> ei.value.code != 43 -> red.
      - Remove the audio_device_gone name-match branch from the
        classifier -> the quoted-name form falls to the generic
        no_audio_device arm -> shown[0].title != "Audio device not
        found" -> red (also caught by
        test_classify_pinned_loopback_gone_names_device_and_points_at_settings).
      - Remove `log.exception("audio capture startup failed")`
        (main.py, commit 6ec44c8) -> no ERROR record carrying exc_info
        -> the caplog assertion below goes red. (This line had no guard
        before; deleting it used to leave the suite green -- a
        behavioural liar.)
    """
    import logging

    from yamabiko import main as _main

    shown = []
    monkeypatch.setattr(
        "yamabiko.ui.error_dialog.show_startup_error",
        lambda err: shown.append(err),
    )

    class _FailingCap:
        def start(self):
            raise RuntimeError("Loopback device not found: 'Speakers (Realtek)'")

    class _App:
        capture = _FailingCap()

    with caplog.at_level(logging.ERROR, logger="yamabiko"):
        with pytest.raises(SystemExit) as ei:
            _main._start_capture_or_exit(_App())

    assert ei.value.code == 43, (
        f"expected sys.exit(43) (STOP_ALL); got exit code {ei.value.code!r}. "
        "The watchdog treats any non-43 exit as a generic crash and "
        "restarts the pipeline (reloading Whisper onto GPU) up to 3x."
    )
    assert len(shown) == 1, (
        f"expected exactly one show_startup_error call; got {len(shown)}"
    )
    assert shown[0].title == "Speakers (Realtek) isn't connected", (
        f"expected the tailored device-gone dialog naming the missing "
        f"device; got title={shown[0].title!r}"
    )
    # The device-failure traceback is logged to the pipeline role log
    # BEFORE the dialog so the failure is triageable (the dialog only
    # carries a short details blob). Assert the observable log record --
    # this is the ONLY guard on main.py's log.exception line.
    rec = next(
        (r for r in caplog.records
         if "audio capture startup failed" in r.getMessage()),
        None,
    )
    assert rec is not None, (
        "expected log.exception('audio capture startup failed'); the "
        "device-failure traceback must reach the pipeline role log for triage"
    )
    assert rec.levelno == logging.ERROR and rec.exc_info is not None, (
        "the capture-failure log must be log.exception (ERROR + exc_info "
        "traceback), not a bare log.error without the traceback"
    )


def test_start_capture_or_exit_passes_through_on_success():
    """Happy-path: _start_capture_or_exit must call capture.start() and
    return normally when no exception is raised. The guard must NOT
    swallow a successful start or call sys.exit spuriously.

    Seeded-defect drill: make _start_capture_or_exit call sys.exit(43)
    unconditionally (even on success) -> SystemExit propagates out of
    this test (which does not catch it) -> error/red.
    """
    from yamabiko import main as _main

    started = []

    class _OkCap:
        def start(self):
            started.append(True)

    class _App:
        capture = _OkCap()

    _main._start_capture_or_exit(_App())  # must NOT raise
    assert started == [True], (
        f"capture.start() was not called or was called more than once; "
        f"started={started!r}"
    )


def test_run_pipeline_delegates_failing_capture_start_to_exit_43(
    _pipeline_env, tmp_path, monkeypatch,
):
    """Closes the wired-not-exercised gap for this seam:
    _start_capture_or_exit is unit-tested DIRECTLY above, and every
    run_pipeline test exercises the SUCCESS path through it — but no
    test proved run_pipeline actually DELEGATES capture startup to the
    guarded helper. A refactor calling `app.capture.start()` bare would
    keep both direct tests green while re-shipping the bug: a
    device-resolution failure propagates as a generic crash, the
    watchdog restarts 3x (reloading Whisper onto the GPU each time),
    and the user gets the generic give-up dialog instead of the
    tailored exit-43 one.

    Driven THROUGH run_pipeline: capture.start() raises the
    device-not-found error; the observable side effects are
    SystemExit(43) escaping run_pipeline + exactly one tailored dialog.

    Seeded-defect drill: replace `_start_capture_or_exit(app)` with
    `app.capture.start()` in run_pipeline -> bare RuntimeError (not
    SystemExit 43) reaches the test -> red.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)

    shown: list = []
    monkeypatch.setattr(
        "yamabiko.ui.error_dialog.show_startup_error",
        lambda err: shown.append(err),
    )

    async def _drive():
        args = env.module.parse_args([])
        app = await env.module.build_pipeline(args)

        def _failing_start():
            raise RuntimeError("Loopback device not found: 'Speakers (Realtek)'")

        app.capture.start = _failing_start
        try:
            await env.module.run_pipeline(app)
        finally:
            # The capture failure fires BEFORE run_pipeline's try/finally,
            # so its teardown never runs; stop the (stubbed) overlay
            # supervisor thread ourselves so it can't outlive the test.
            stop = getattr(app, "overlay_stop", None)
            if stop is not None:
                stop.set()

    with pytest.raises(SystemExit) as ei:
        asyncio.run(_drive())

    assert ei.value.code == 43, (
        f"run_pipeline must route a failing capture.start() through "
        f"_start_capture_or_exit's tailored exit-43 path; got "
        f"{ei.value.code!r} (a non-43 exit means the watchdog restarts "
        f"the pipeline 3x before a generic give-up dialog)"
    )
    assert len(shown) == 1, (
        f"expected exactly one tailored startup dialog; got {len(shown)}"
    )
    assert shown[0].title == "Speakers (Realtek) isn't connected", (
        f"expected the tailored device-gone dialog (the injected failure "
        f"carries a quoted device name); got title={shown[0].title!r}"
    )


def test_build_pipeline_exits_43_on_asr_startup_failure(
    _pipeline_env, tmp_path, monkeypatch,
):
    """ASR initialization failure (missing CUDA DLL, missing model
    files, disk-full, ...) must:
      - classify the exception via classify_startup_exception
      - surface a tailored StartupError via show_startup_error
      - exit with code 43 (the sentinel the watchdog uses to
        suppress its generic "keeps crashing" dialog)

    Without the 43 sentinel, a user with a failing GPU sees a generic
    "The pipeline exited after only X seconds" dialog with no
    actionable details. With 43, they see "No NVIDIA graphics card
    found" or "Some model files are missing" with what to try.

    Seeded-defect check: change `sys.exit(43)` in the ASR except
    block to `return None` or `sys.exit(1)` -> either the test sees
    no SystemExit, or the exit code is wrong. Both fail this test.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)

    def _asr_raises(cfg):
        raise RuntimeError("CUDA not available")
    monkeypatch.setattr(env.module, "build_asr", _asr_raises)

    from yamabiko.ui import error_dialog
    shown_errors: list = []
    monkeypatch.setattr(
        error_dialog, "show_startup_error",
        lambda e: shown_errors.append(e),
    )

    args = env.module.parse_args([])
    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(env.module.build_pipeline(args))

    assert exc_info.value.code == 43, (
        f"ASR init failure did not exit 43; got {exc_info.value.code}. "
        f"Watchdog will layer a generic crash-loop dialog on top of "
        f"the tailored ASR error."
    )
    assert len(shown_errors) == 1, (
        f"show_startup_error was called {len(shown_errors)} times; "
        f"expected exactly 1"
    )
    assert isinstance(shown_errors[0], error_dialog.StartupError), (
        f"show_startup_error received a {type(shown_errors[0]).__name__}, "
        f"not a StartupError; classify_startup_exception wiring broken"
    )


def test_build_pipeline_exits_43_on_translator_sanity_check_failure(
    _pipeline_env, tmp_path, monkeypatch,
):
    """Translator sanity-check failure (llama-server down, refused
    connection, bad auth) must also surface a tailored dialog +
    exit 43. Same contract as ASR failure.

    Seeded-defect check: change `sys.exit(43)` in the translator
    except block to `raise` (re-raise the exception) -> the outer
    asyncio.run surfaces the exception, SystemExit isn't raised
    with code 43, this test fails.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)

    class _FailingTranslator(_FakeTranslator):
        async def sanity_check(self, genre):
            raise ConnectionError("llama-server refused connection")
    monkeypatch.setattr(env.module, "Translator", _FailingTranslator)

    from yamabiko.ui import error_dialog
    shown_errors: list = []
    monkeypatch.setattr(
        error_dialog, "show_startup_error",
        lambda e: shown_errors.append(e),
    )

    args = env.module.parse_args([])
    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(env.module.build_pipeline(args))

    assert exc_info.value.code == 43, (
        f"translator sanity-check failure did not exit 43; "
        f"got {exc_info.value.code}"
    )
    assert len(shown_errors) == 1
    assert isinstance(shown_errors[0], error_dialog.StartupError)


def test_build_pipeline_exits_43_naming_file_when_silero_cache_corrupt(
    _pipeline_env, tmp_path, monkeypatch,
):
    """A cached Silero ONNX that fails its pinned digest must surface a
    tailored dialog naming the bad file, then exit 43. Any other exit makes
    the watchdog restart the pipeline into the same failure until it gives
    up with a generic dialog, and re-running the wizard cannot repair the
    file (hub reuses the cached file for a pinned commit).

    The real digest check runs against a file of the wrong bytes.
    Seeded defect: construct SileroChunker outside the classify/dialog/
    exit-43 pattern -> the RuntimeError propagates instead of
    SystemExit(43) -> red.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)

    import yamabiko.vad.silero_chunker as silero_mod
    # The path _pipeline_env's _default_model_path stub resolves to.
    bad_model = tmp_path / "fake-silero-vad.onnx"
    bad_model.write_bytes(b"not the pinned silero graph")

    def _digest_checked_vad(model_path, sample_rate=16000):
        silero_mod._verify_model_artifact(model_path)
        return _FakeSileroVAD(model_path, sample_rate)
    monkeypatch.setattr(silero_mod, "SileroVAD", _digest_checked_vad)

    from yamabiko.ui import error_dialog
    shown_errors: list = []
    monkeypatch.setattr(
        error_dialog, "show_startup_error",
        lambda e: shown_errors.append(e),
    )

    args = env.module.parse_args([])
    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(env.module.build_pipeline(args))

    assert exc_info.value.code == 43, (
        f"corrupt Silero cache did not exit 43; got {exc_info.value.code!r}. "
        f"The watchdog restarts any other exit into the same failure."
    )
    assert len(shown_errors) == 1, (
        f"expected exactly one show_startup_error call; got {len(shown_errors)}"
    )
    assert shown_errors[0].title == "A model file is damaged", (
        f"expected the corrupt-model-file dialog; got "
        f"title={shown_errors[0].title!r}"
    )
    assert shown_errors[0].message == f"{bad_model.name} didn't pass its check.", (
        f"the dialog must name the file to delete; got "
        f"message={shown_errors[0].message!r}"
    )


def test_pipeline_app_on_stage_escalate_arms_restart_flag_and_sets_done(
    _pipeline_env, tmp_path,
):
    """PipelineApp.on_stage_escalate must arm stage_escalation_requested
    AND set the done event, so the pipeline exits EXIT_STAGE_ESCALATION
    (45, RESTART) instead of 0 (STOP_SELF). With `done` alone, amain
    returned 0, classify(0) == STOP_SELF, and the watchdog read a stage
    crash-loop (e.g. translator dead: 10 failures in 60 s) as a clean
    quit -- cascading a full stop with no dialog and no restart. Mirrors
    request_stream_death.

    The method is extracted from a local closure inside run_pipeline
    so this test can invoke it directly (charter 3.9: invoke the
    handler, don't just assert it's wired).

    Seeded-defect checks (one line each):
      - remove `self.stage_escalation_requested = True` from the method
        body -> flag assertion red (escalation exits 0 again; silent
        death regression).
      - remove `self.done.set()` -> done assertion red (crash-looping
        stages keep cycling with no shutdown).
    """
    env = _pipeline_env
    _write_overrides(tmp_path)

    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))

    assert not app.done.is_set(), (
        "precondition: done must start unset"
    )
    assert app.stage_escalation_requested is False, (
        "precondition: stage_escalation_requested must start False"
    )

    app.on_stage_escalate("asr")
    assert app.stage_escalation_requested is True, (
        "PipelineApp.on_stage_escalate did not arm stage_escalation_requested; "
        "amain would return 0 (STOP_SELF) and the watchdog would read the "
        "stage crash-loop as a clean quit -- silent death, no restart"
    )
    assert app.done.is_set(), (
        "PipelineApp.on_stage_escalate did not set done; a stage "
        "crossing the escalation threshold will not shut the "
        "pipeline down"
    )


def test_run_pipeline_wires_stages_to_app_escalate_callback(
    _pipeline_env, tmp_path, monkeypatch,
):
    """run_pipeline must pass app.on_stage_escalate as the
    on_escalate kwarg to asr_stage and translation_stage. A regression
    that passed a no-op lambda (or nothing) would leave crash-loops
    with no shutdown path.

    The observable assertion: done becomes set when ANY of the
    stage on_escalate callbacks fires. We capture the kwargs passed
    to asr_stage and translation_stage and verify on_escalate is
    the app method.

    Seeded-defect check: change run_pipeline to pass
    `on_escalate=lambda stage: None`, or pass no on_translated ->
    captured kwarg is a different callable (or absent), assertion fails.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)

    asr_stage_kwargs: list = []
    translation_stage_kwargs: list = []

    # Wrap the real asr_stage / translation_stage to capture kwargs,
    # then return a coroutine that returns immediately (so the tasks
    # complete without doing pipeline work).
    async def _capture_asr_stage(*args, **kwargs):
        asr_stage_kwargs.append(kwargs)

    async def _capture_translation_stage(*args, **kwargs):
        translation_stage_kwargs.append(kwargs)

    monkeypatch.setattr(env.module, "asr_stage", _capture_asr_stage)
    monkeypatch.setattr(
        env.module, "translation_stage", _capture_translation_stage,
    )

    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert len(asr_stage_kwargs) == 1, (
        "asr_stage should have been started exactly once"
    )
    assert len(translation_stage_kwargs) == 1, (
        "translation_stage should have been started exactly once"
    )
    # The on_escalate kwarg must be app.on_stage_escalate (bound
    # method equality). A regression wiring a lambda or None would
    # make this comparison fail.
    assert asr_stage_kwargs[0]["on_escalate"] == app.on_stage_escalate, (
        "asr_stage.on_escalate is not app.on_stage_escalate; "
        "supervisor escalations won't reach the shutdown path"
    )
    assert translation_stage_kwargs[0]["on_escalate"] == app.on_stage_escalate, (
        "translation_stage.on_escalate is not app.on_stage_escalate"
    )
    assert translation_stage_kwargs[0]["on_translated"] == app.note_translation_ok, (
        "translation_stage.on_translated is not app.note_translation_ok: a "
        "completed translation would never clear the translator problem"
    )


def test_run_pipeline_threads_series_context_hint_flag_into_translation_stage(
    _pipeline_env, tmp_path, monkeypatch,
):
    """`translation.series_context_hint: true` in the merged config must
    arrive at translation_stage as series_hint_enabled=True. This is the only
    test crossing the cfg-key -> stage-kwarg seam: with a wrong key read the
    flag is silently always-False and the feature is dead even for opted-in
    users (the stage-level hint tests inject the flag directly, so they
    cannot see it). The Series-line behavior behind the flag is asserted on
    the composed user message in test_main_pipeline.py.

    Seeded defect (one line): read a wrong key in build_pipeline
    (e.g. `series_context_hints`) -> the captured kwarg is False -> red.
    Also red if run_pipeline stops forwarding the kwarg.
    """
    env = _pipeline_env
    _write_overrides(tmp_path, {"translation": {"series_context_hint": True}})

    translation_stage_kwargs: list = []

    async def _capture_translation_stage(*args, **kwargs):
        translation_stage_kwargs.append(kwargs)

    monkeypatch.setattr(
        env.module, "translation_stage", _capture_translation_stage,
    )

    asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert len(translation_stage_kwargs) == 1, (
        "translation_stage should have been started exactly once"
    )
    assert translation_stage_kwargs[0].get("series_hint_enabled") is True, (
        "translation.series_context_hint=true did not reach translation_stage "
        "as series_hint_enabled=True; the config key has no live consumer "
        f"(got {translation_stage_kwargs[0].get('series_hint_enabled')!r})"
    )


def test_run_pipeline_defaults_series_hint_off_at_the_translation_stage(
    _pipeline_env, tmp_path, monkeypatch,
):
    """With NO translation override (the shipped default in force),
    translation_stage receives series_hint_enabled=False. This pins the
    OFF default across the whole shipped-YAML -> build -> run seam: the
    un-A/B'd lever ships silently if the default ever flips, and the
    opt-IN test above cannot see that (it writes the override).

    Seeded defect (one line): flip default_config.yaml's
    `series_context_hint:` to true -> the captured kwarg is True -> red.
    Also red if build_pipeline hardcodes the flag on.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)  # no translation key: shipped default rules

    translation_stage_kwargs: list = []

    async def _capture_translation_stage(*args, **kwargs):
        translation_stage_kwargs.append(kwargs)

    monkeypatch.setattr(
        env.module, "translation_stage", _capture_translation_stage,
    )

    asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert len(translation_stage_kwargs) == 1, (
        "translation_stage should have been started exactly once"
    )
    assert translation_stage_kwargs[0].get("series_hint_enabled") is False, (
        "the shipped default must arrive at translation_stage as "
        "series_hint_enabled=False (A/B gate pending); got "
        f"{translation_stage_kwargs[0].get('series_hint_enabled')!r}"
    )


def test_build_pipeline_resolves_series_hint_false_when_the_key_is_absent(
    _pipeline_env, tmp_path,
):
    """A config carrying NO translation.series_context_hint key at all
    (the --config escape hatch reads the file verbatim, no defaults
    merge) resolves to app.series_hint_enabled=False: build_pipeline's
    .get fallback is the OFF default's last line of defense, and the
    shipped-YAML pin cannot see it (the key is present there).

    Seeded defect (one line): flip build_pipeline's
    `.get("series_context_hint", False)` fallback to True -> red. (The
    shipped-YAML default can't mask it: the key is absent here.)
    """
    import yaml as _yaml

    from yamabiko import paths as _paths

    env = _pipeline_env
    cfg = _yaml.safe_load(
        _paths.shipped_defaults_path().read_text(encoding="utf-8")
    )
    cfg["translation"].pop("series_context_hint", None)
    cfg["romanization"]["enabled"] = False
    cfg["hotkeys"]["enabled"] = False
    cfg["overlay"]["browser"] = False
    explicit = tmp_path / "no-series-hint-key.yaml"
    explicit.write_text(
        _yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8",
    )

    args = env.module.parse_args(["--config", str(explicit)])
    app = asyncio.run(env.module.build_pipeline(args))

    assert app.series_hint_enabled is False, (
        "an absent translation.series_context_hint key must resolve to "
        "series_hint_enabled=False (the .get fallback IS the shipped OFF "
        f"default here); got {app.series_hint_enabled!r}"
    )


def test_build_pipeline_falls_back_when_glossaries_dir_override_is_nonexistent_relative(
    _pipeline_env, tmp_path,
):
    """When cfg['romanization']['glossaries_dir'] is a RELATIVE path
    that doesn't exist (relative to cwd), build_pipeline must fall
    back to the user-writable default `paths.app_data_dir() / "glossaries"`
    (seeded from the bundle) rather than the bad override path.

    Context: the legacy default `"yamabiko/glossaries"` in shipped
    default_config.yaml was a cwd-dependent relative path that broke in
    the frozen exe (cwd = $INSTDIR, glossaries live under
    $INSTDIR\\_internal\\yamabiko\\glossaries). ANY relative override is
    now rejected (cwd differs between dev and the installed app), so a
    non-existent one falls back too. The paired
    test_build_pipeline_falls_back_when_glossaries_dir_override_is_existing_relative
    proves an EXISTING relative path is rejected as well.

    Seeded-defect check: remove the fallback branch
        if not gdir.is_absolute():
            gdir = default_gdir
    -> GlossaryManager receives Path("does/not/exist") instead of
    the default, this test fails.
    """
    from yamabiko import paths

    env = _pipeline_env
    _write_overrides(tmp_path, {
        "romanization": {
            "enabled": True,
            "glossaries_dir": "does/not/exist",  # relative, not real
        },
    })

    args = env.module.parse_args([])
    asyncio.run(env.module.build_pipeline(args))

    assert len(env.glossary_instances) == 1
    resolved_gdir = env.glossary_instances[0].gdir
    # The default is the user-WRITABLE app_data dir: learned proper-nouns
    # must survive upgrades, so they can't live in the read-only bundle
    # that File /r re-extracts on every install.
    expected_default = paths.app_data_dir() / "glossaries"

    assert resolved_gdir == expected_default, (
        f"relative-nonexistent override did not fall back to bundled "
        f"default {expected_default!r}; got {resolved_gdir!r}. "
        f"A user with a bad glossaries_dir would silently run with "
        f"no romanization rather than the shipped defaults."
    )


async def _build_pipeline_app(env, tmp_path, extra_overrides: dict | None = None):
    """Shared helper: build a PipelineApp with hotkeys on + browser on
    + optional extra overrides. Drives run_pipeline with early-done so
    the browser-server branch populates app.manager, then returns the
    finished app for method-level assertions.
    """
    overrides = {"hotkeys": {"enabled": True}, "overlay": {"browser": True}}
    if extra_overrides:
        overrides.update(extra_overrides)
    _write_overrides(tmp_path, overrides)
    args = env.module.parse_args([])
    app = await env.module.build_pipeline(args)
    app.done.set()
    await env.module.run_pipeline(app)
    return app


def test_handle_pause_hotkey_toggles_and_broadcasts_when_manager_wired(
    _pipeline_env, tmp_path,
):
    """PipelineApp.handle_pause_hotkey must:
      1. Toggle app_state.paused (False -> True on first press).
      2. Broadcast a state event with exactly the right shape:
         {"type": "state", "genre": <current>, "paused": <new>}.

    The closure was previously locked inside run_pipeline and never
    invoked by any test. Extracting to PipelineApp.handle_pause_hotkey
    makes the body directly unit-testable.

    Seeded-defect checks:
      - Mutant 366: flip `if app.manager is not None` -> broadcast
        never fires, len(broadcasts) == 0, this test fails.
      - Mutant 367: set `snap = None` -> `snap["genre"]` raises
        TypeError inside handle_pause_hotkey, test fails.
      - Mutant 368: change dict key `"type"` -> broadcast payload
        lacks the expected key, this test fails.
    """
    env = _pipeline_env
    app = asyncio.run(_build_pipeline_app(env, tmp_path))

    assert app.manager is env.fake_fastapi_app.state.manager, (
        "precondition: app.manager must be the fixture's fake manager"
    )
    assert not app.app_state.paused, "precondition: paused starts False"

    asyncio.run(app.handle_pause_hotkey())

    # State toggled.
    assert app.app_state.paused, (
        "handle_pause_hotkey did not flip paused=False -> True"
    )
    # Exactly one broadcast with the right payload.
    bc = app.manager.broadcasts
    assert len(bc) == 1, (
        f"expected exactly one broadcast; got {len(bc)}: {bc!r}"
    )
    payload = bc[0]
    assert payload.get("type") == "state", (
        f"broadcast payload missing type='state'; got {payload!r}"
    )
    assert payload.get("paused") is True, (
        f"broadcast payload must report paused=True after first press; "
        f"got {payload!r}"
    )
    assert payload.get("genre") == app.app_state.genre, (
        f"broadcast payload genre {payload.get('genre')!r} "
        f"doesn't match app_state.genre {app.app_state.genre!r}"
    )


def test_handle_pause_hotkey_toggles_state_without_broadcast_when_no_manager(
    _pipeline_env, tmp_path,
):
    """Negative paired case: when app.manager is None (the state every
    PipelineApp is in until run_pipeline wires the WS server — and
    permanently if the server fails to build), handle_pause_hotkey
    still toggles state but skips the broadcast.

    Without this guard, invoking the hotkey with no manager would
    `None.broadcast(...)` -> AttributeError. Protects the
    `if self.manager is not None` check that mutant 366 targeted.

    (Note: this used to manufacture manager=None via overlay.browser=False;
    the server now starts unconditionally, so the un-run build_pipeline
    output IS the no-manager state.)
    """
    env = _pipeline_env
    _write_overrides(tmp_path, {"hotkeys": {"enabled": True}})

    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))
    assert app.manager is None, (
        "precondition: build_pipeline must leave manager unwired"
    )

    asyncio.run(app.handle_pause_hotkey())
    assert app.app_state.paused, "state did not toggle when manager is None"
    # No broadcast attempted because there's no manager.


def test_handle_pause_hotkey_second_invocation_resumes(
    _pipeline_env, tmp_path,
):
    """Two presses in sequence: False -> True -> False. Broadcasts
    twice with the updated paused flag each time.

    Protects the `if self.app_state.paused: resume() else: pause()`
    branch from mutations that invert the condition (flip would
    break the resume path; first-press-pause path is covered by the
    test above).

    Seeded-defect check: flip the if/else branches in handle_pause_hotkey
    -> second invocation stays True (or goes back to False after
    the first also went False) -> assertion fails.
    """
    env = _pipeline_env
    app = asyncio.run(_build_pipeline_app(env, tmp_path))

    asyncio.run(app.handle_pause_hotkey())
    asyncio.run(app.handle_pause_hotkey())

    assert not app.app_state.paused, (
        f"two presses should end at paused=False; got {app.app_state.paused}"
    )
    assert len(app.manager.broadcasts) == 2, (
        f"expected 2 broadcasts (one per press); got "
        f"{len(app.manager.broadcasts)}"
    )
    # Specifically: last broadcast reports the resumed state.
    assert app.manager.broadcasts[-1].get("paused") is False, (
        f"final broadcast should report paused=False; got "
        f"{app.manager.broadcasts[-1]!r}"
    )


def test_pause_hotkey_state_frame_carries_series(_pipeline_env, tmp_path):
    """The pause hotkey used to hand-build its {"type":"state",...} frame
    with only genre+paused. ws_client emitted seriesReceived for every state
    frame, so the missing series keys were read as 'series cleared' and the
    tray's Series line reset to (none) on every Ctrl+Alt+P -- while the
    pipeline gazetteer stayed active. Routing handle_pause_hotkey's
    broadcast through AppState.state_frame() makes the pause frame carry the
    active series (the producer-side fix), so the tray line survives a pause.

    Seeded-defect check: revert handle_pause_hotkey to the two-key
    hand-built {"type","genre","paused"} dict -> series_media_id absent from
    the broadcast -> KeyError on the assert -> red.
    """
    env = _pipeline_env
    app = asyncio.run(_build_pipeline_app(env, tmp_path))
    app.app_state.set_series(7, "Frieren (2023)")

    asyncio.run(app.handle_pause_hotkey())

    frame = app.manager.broadcasts[-1]
    assert frame["series_media_id"] == 7, (
        f"pause frame dropped the active series media_id; got {frame!r}"
    )
    assert frame["series_label"] == "Frieren (2023)", (
        f"pause frame dropped the active series label; got {frame!r}"
    )


def test_handle_toggle_overlay_hotkey_broadcasts_when_manager_wired(
    _pipeline_env, tmp_path,
):
    """PipelineApp.handle_toggle_overlay_hotkey broadcasts the
    toggle-overlay command via make_toggle_overlay_cmd() to any
    connected overlay.

    The command shape is owned by yamabiko.display.broadcast; this
    test just asserts that the broadcast fires and the payload
    matches what make_toggle_overlay_cmd returns. A regression that
    broadcast something different would fail the key match.
    """
    env = _pipeline_env
    app = asyncio.run(_build_pipeline_app(env, tmp_path))

    from yamabiko.display.broadcast import make_toggle_overlay_cmd
    expected = make_toggle_overlay_cmd()

    asyncio.run(app.handle_toggle_overlay_hotkey())

    bc = app.manager.broadcasts
    assert len(bc) == 1, f"expected one broadcast; got {bc!r}"
    assert bc[0] == expected, (
        f"broadcast payload {bc[0]!r} != expected {expected!r}. "
        f"Either handle_toggle_overlay_hotkey sent the wrong command "
        f"or make_toggle_overlay_cmd's shape drifted."
    )


def test_handle_toggle_overlay_hotkey_skips_when_no_manager(
    _pipeline_env, tmp_path,
):
    """Paired negative: no manager => return without error (and
    without a broadcast). Previously enforced via a local closure;
    now enforced via the method's guard.

    Seeded-defect check: drop the `if self.manager is None: return`
    early-out -> None.broadcast(...) raises AttributeError, this
    test fails.

    (Note: manager=None now comes from the un-run build_pipeline output,
    not from overlay.browser=False — the server starts unconditionally.)
    """
    env = _pipeline_env
    _write_overrides(tmp_path, {"hotkeys": {"enabled": True}})

    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))
    assert app.manager is None, (
        "precondition: build_pipeline must leave manager unwired"
    )

    # Must not raise.
    asyncio.run(app.handle_toggle_overlay_hotkey())


def test_handle_show_series_hotkey_broadcasts_show_series_palette_when_manager_wired(
    _pipeline_env, tmp_path,
):
    """PipelineApp.handle_show_series_hotkey broadcasts exactly the
    show-series-palette command (via make_show_series_cmd) to any connected
    overlay. Mirrors the toggle_overlay hotkey-body test.

    Seeded defect: have the method broadcast make_toggle_overlay_cmd() (or
    change make_show_series_cmd's payload) -> the exact-match assert goes red.
    """
    env = _pipeline_env
    app = asyncio.run(_build_pipeline_app(env, tmp_path))

    asyncio.run(app.handle_show_series_hotkey())

    bc = app.manager.broadcasts
    assert len(bc) == 1, f"expected one broadcast; got {bc!r}"
    assert bc[0] == {"type": "show_series_palette"}, (
        f"broadcast payload {bc[0]!r} != expected show_series_palette command"
    )


def test_handle_show_series_hotkey_skips_when_no_manager(_pipeline_env, tmp_path):
    """Paired negative: no manager => return without error and without a
    broadcast (the state build_pipeline leaves the app in until run_pipeline
    wires the WS server).

    Seeded-defect check: drop the `if self.manager is None: return` early-out
    -> None.broadcast(...) raises AttributeError -> this test fails.
    """
    env = _pipeline_env
    _write_overrides(tmp_path, {"hotkeys": {"enabled": True}})

    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))
    assert app.manager is None, (
        "precondition: build_pipeline must leave manager unwired"
    )

    # Must not raise.
    asyncio.run(app.handle_show_series_hotkey())


def test_hotkey_conflict_before_any_client_connects_is_delivered_to_first_client(
    _pipeline_env, tmp_path, monkeypatch,
):
    """A combo already taken by another program used to be only logged: the
    conflict fires inside HotkeyManager.register, before the WS server exists,
    so no client could be told. The notice must wait on AppState and reach the
    first client right after its snapshot frames.

    The fake register raises the conflict through the on_conflict callback
    run_pipeline wires (same timing as production: manager not built yet);
    the real /ws handler then serves the first client.

    Seeded defects: drop the pending-notice send from the /ws handler, or the
    queue_notice call in notify_hotkey_conflict -> the first client's third
    frame is the resume barrier's state broadcast, not the notice -> red.
    """
    from fastapi.testclient import TestClient

    import yamabiko.hotkeys as hk_mod
    from yamabiko.display.browser_server import build_app as real_build_app

    env = _pipeline_env
    _write_overrides(tmp_path, {"hotkeys": {"enabled": True}})

    def _conflicting_register(**kwargs):
        env.fake_hotkey_mgr.register_calls.append(kwargs)
        kwargs["on_conflict"](kwargs["pause"], hk_mod.HotkeyTaken(kwargs["pause"]))

    monkeypatch.setattr(env.fake_hotkey_mgr, "register", _conflicting_register)

    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    server = real_build_app(state=app.app_state, bound_port=8765, ws_token="tok")
    client = TestClient(server)
    with client.websocket_connect("/ws?t=tok") as ws:
        assert ws.receive_json()["type"] == "state"
        assert ws.receive_json()["type"] == "style"
        assert ws.receive_json()["type"] == "problems"
        # Barrier: the resume broadcast is sent AFTER any queued notice, so
        # a missing notice shows up as this state frame instead of a hang.
        ws.send_json({"type": "command", "action": "resume"})
        frame = ws.receive_json()

    assert frame == {
        "type": "notice",
        "title": "Shortcut unavailable",
        "body": "Ctrl+Alt+P is used by another app. Pick a new one in Settings.",
        "settings_page": "shortcuts",
    }, f"first client did not receive the shortcut notice; got {frame!r}"


def test_a_shortcut_notice_waits_while_only_a_browser_is_connected(
    _pipeline_env, tmp_path,
):
    """A browser/OBS page can't show notices, so a shortcut conflict while
    only a browser is connected waits on AppState for the Qt overlay instead
    of being broadcast.

    Seeded defect: check client_count() instead of native_client_count() in
    notify_hotkey_conflict -> the notice is dispatched, not queued -> red."""
    env = _pipeline_env
    _write_overrides(tmp_path)
    app = asyncio.run(env.module.build_pipeline(env.module.parse_args([])))
    app.manager = _FakeFastAPIManager()
    app.manager.clients = 1

    app.notify_hotkey_conflict("ctrl+alt+p", RuntimeError("x"))

    assert len(app.app_state.take_pending_notices()) == 1


def test_apply_hotkeys_rebinds_the_pipeline_with_the_new_combos(_pipeline_env, tmp_path):
    """Settings' set_hotkeys lands in PipelineApp.apply_hotkeys: the config
    takes the new combo and the manager is cleared, then bound again with it.

    Seeded defects: update cfg but never re-register, or re-register without
    clearing first -> red."""
    env = _pipeline_env
    _write_overrides(tmp_path, {"hotkeys": {"enabled": True}})
    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))
    env.fake_hotkey_mgr.register_calls.clear()
    env.fake_hotkey_mgr.calls.clear()

    app.apply_hotkeys({"pause_resume": "Ctrl+Alt+K"})

    assert app.cfg["hotkeys"]["pause_resume"] == "Ctrl+Alt+K"
    assert env.fake_hotkey_mgr.calls == ["unregister", "register"]
    assert [call["pause"] for call in env.fake_hotkey_mgr.register_calls] == ["Ctrl+Alt+K"]


def test_apply_hotkeys_announces_only_changed_or_resent_keys(_pipeline_env, tmp_path, monkeypatch):
    """A shortcut Settings did not touch stays quiet: apply_hotkeys announces a
    conflict only for the keys in its payload, and a reconnect replay
    announces only keys whose combo changed. Every other conflict is logged,
    not announced. Another app holds the default pause combo Ctrl+Alt+P and
    Ctrl+Alt+K throughout.

    Seeded defects: announce every conflict on each apply, announce the
    unchanged keys of a replay, or announce nothing on a replay -> red."""
    import yamabiko.hotkeys as hk_mod
    env = _pipeline_env
    _write_overrides(tmp_path, {"hotkeys": {"enabled": True}})
    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    held = {"Ctrl+Alt+P", "Ctrl+Alt+K"}

    def _register(**kwargs):
        env.fake_hotkey_mgr.calls.append("register")
        env.fake_hotkey_mgr.register_calls.append(kwargs)
        for combo in (kwargs["pause"], kwargs["toggle_overlay"], kwargs["show_series"]):
            if combo in held:
                kwargs["on_conflict"](combo, hk_mod.HotkeyTaken(combo))

    monkeypatch.setattr(env.fake_hotkey_mgr, "register", _register)
    announced = []
    monkeypatch.setattr(app, "notify_hotkey_conflict",
                        lambda combo, exc: announced.append(combo))

    # Only toggle_overlay changed: the held pause combo is logged, not announced.
    app.apply_hotkeys({"toggle_overlay": "Ctrl+Alt+J"})
    assert announced == []

    # The user sent pause_resume itself: its conflict is announced.
    app.apply_hotkeys({"pause_resume": "Ctrl+Alt+P"})
    assert announced == ["Ctrl+Alt+P"]

    # A reconnect replay of the same, unchanged combo stays quiet.
    app.apply_hotkeys({"pause_resume": "Ctrl+Alt+P"}, replay=True)
    assert announced == ["Ctrl+Alt+P"]

    # A replay of a changed combo announces it.
    app.apply_hotkeys({"toggle_overlay": "Ctrl+Alt+K"}, replay=True)
    assert announced == ["Ctrl+Alt+P", "Ctrl+Alt+K"]

    # Both keys in the payload: both conflicts are announced, in binding order.
    app.apply_hotkeys({"pause_resume": "Ctrl+Alt+P", "toggle_overlay": "Ctrl+Alt+K"})
    assert announced == ["Ctrl+Alt+P", "Ctrl+Alt+K", "Ctrl+Alt+P", "Ctrl+Alt+K"]


def test_a_restarted_pipeline_keeps_quiet_about_shortcuts_it_already_announced(
    _pipeline_env, tmp_path, monkeypatch,
):
    """Decision 72: a restart exit carries the shortcuts the pipeline
    announced as taken, and the next pipeline's boot announces a conflict
    only for a key whose combo differs from the carried one. Another app
    holds Ctrl+Alt+P, Ctrl+Alt+K and Ctrl+Alt+J.

    Seeded defects: leave the shortcuts out of the carry, never read the
    carried variable, or skip a changed key at boot -> red."""
    import yamabiko.hotkeys as hk_mod
    from yamabiko import problems

    env = _pipeline_env
    held = {"Ctrl+Alt+P", "Ctrl+Alt+K", "Ctrl+Alt+J"}

    def _register(**kwargs):
        env.fake_hotkey_mgr.register_calls.append(kwargs)
        for combo in (kwargs["pause"], kwargs["toggle_overlay"], kwargs["show_series"]):
            if combo in held:
                kwargs["on_conflict"](combo, hk_mod.HotkeyTaken(combo))

    monkeypatch.setattr(env.fake_hotkey_mgr, "register", _register)
    announced = []
    monkeypatch.setattr(env.module.PipelineApp, "notify_hotkey_conflict",
                        lambda self, combo, exc: announced.append(combo))

    # A fresh launch announces both taken shortcuts; its restart exit carries them.
    _write_overrides(tmp_path, {"hotkeys": {"enabled": True, "toggle_overlay": "Ctrl+Alt+K"}})
    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))
    assert announced == ["Ctrl+Alt+P", "Ctrl+Alt+K"]
    app.on_stage_escalate("asr")
    asyncio.run(app.announce_restart_problems())
    carry = problems.take_carry()
    assert carry == problems.Carry((), False, {"pause_resume": "Ctrl+Alt+P",
                                               "toggle_overlay": "Ctrl+Alt+K"})

    # The next pipeline (toggle_overlay moved to Ctrl+Alt+J meanwhile) stays
    # quiet about the unchanged pause shortcut and announces the changed one.
    monkeypatch.setenv(problems.CARRY_HOTKEYS_ENV, problems.carry_hotkeys_env_value(carry.hotkeys))
    _write_overrides(tmp_path, {"hotkeys": {"enabled": True, "toggle_overlay": "Ctrl+Alt+J"}})
    announced.clear()
    app2 = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))
    assert announced == ["Ctrl+Alt+J"]
    assert app2.announced_hotkeys == {"pause_resume": "Ctrl+Alt+P",
                                      "toggle_overlay": "Ctrl+Alt+J"}


def test_dispatch_hotkey_swallows_closed_loop_and_closes_coro():
    """Invocation test: a keystroke arriving after the loop closed must
    not raise out of the keyboard daemon thread, and the un-scheduled coroutine
    must be closed (no 'coroutine was never awaited' ResourceWarning).

    Seeded-defect check: drop the try/except in PipelineApp.dispatch_hotkey ->
    asyncio.run_coroutine_threadsafe(coro, closed_loop) raises
    RuntimeError('Event loop is closed') -> this test fails (unexpected exception).
    """
    import inspect
    import types

    from yamabiko.main import PipelineApp

    loop = asyncio.new_event_loop()
    loop.close()
    fake_self = types.SimpleNamespace(loop=loop)

    async def _coro():
        return None

    coro = _coro()
    # Unbound call: dispatch_hotkey only reads self.loop, so a minimal
    # SimpleNamespace stub is sufficient and keeps this a pure unit test.
    PipelineApp.dispatch_hotkey(fake_self, coro)  # must not raise

    assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED, (
        "dispatch_hotkey must close the un-scheduled coroutine to suppress "
        "'coroutine was never awaited' ResourceWarning"
    )


def test_dispatch_hotkey_schedules_coro_on_open_loop():
    """Happy-path pair: with a RUNNING loop, dispatch_hotkey actually
    schedules the coroutine on it (not just swallows the closed-loop case).

    Seeded-defect check: replace the asyncio.run_coroutine_threadsafe(...) call
    in PipelineApp.dispatch_hotkey with `pass` -> the coroutine never runs ->
    the threading.Event never sets -> this test fails on the 2 s wait.
    """
    import threading
    import types

    from yamabiko.main import PipelineApp

    ran = threading.Event()

    async def _coro():
        ran.set()

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    try:
        PipelineApp.dispatch_hotkey(types.SimpleNamespace(loop=loop), _coro())
        assert ran.wait(timeout=2.0), (
            "dispatch_hotkey did not schedule the coroutine on the open loop"
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2.0)
        loop.close()


def test_seed_user_glossaries_copies_if_absent_and_preserves_learned(tmp_path, monkeypatch):
    """Per-file copy-if-absent: first run seeds shipped files; a file that
    already holds learned entries is never overwritten; and a newly-shipped
    seed file reaches an existing install on a later run.

    Seeded defect: drop the `not target.exists()` check in
    _seed_user_glossaries -> the re-seed overwrites the learned anime.json
    -> this fails.
    """
    import yamabiko.main as m
    from yamabiko import paths

    src = tmp_path / "bundle" / "glossaries"
    src.mkdir(parents=True)
    (src / "anime.json").write_text('{"version": 1, "entries": {}}', encoding="utf-8")
    monkeypatch.setattr(paths, "bundle_data_dir", lambda: tmp_path / "bundle")

    dest = tmp_path / "appdata" / "glossaries"
    m._seed_user_glossaries(dest)
    assert (dest / "anime.json").is_file(), "first run did not seed the glossaries"

    # A learned entry lands in the writable copy; a NEW seed file ships.
    (dest / "anime.json").write_text(
        '{"version": 1, "entries": {"learned": 1}}', encoding="utf-8",
    )
    (src / "news.json").write_text('{"version": 1, "entries": {}}', encoding="utf-8")

    m._seed_user_glossaries(dest)
    assert '"learned"' in (dest / "anime.json").read_text(encoding="utf-8"), (
        "re-seed clobbered the learned entry"
    )
    assert (dest / "news.json").is_file(), (
        "a newly-shipped seed file did not reach the existing install"
    )


def test_seed_user_glossaries_skips_underscore_files(tmp_path, monkeypatch):
    """_seed.yaml (and _denylist.txt) are read in place from the
    read-only bundle — copying them into app-data froze them at
    first-install version. Only genre JSONs and user.yaml belong in the
    writable dir.

    Seeded defect: drop the underscore-prefix skip in
    _seed_user_glossaries -> _seed.yaml lands in dest -> red.
    """
    import yamabiko.main as m
    from yamabiko import paths

    src = tmp_path / "bundle" / "glossaries"
    src.mkdir(parents=True)
    (src / "anime.json").write_text('{"version": 1, "entries": {}}', encoding="utf-8")
    (src / "user.yaml").write_text("entries: {}\n", encoding="utf-8")
    (src / "_seed.yaml").write_text("entries: {}\n", encoding="utf-8")
    (src / "_denylist.txt").write_text("\n", encoding="utf-8")
    monkeypatch.setattr(paths, "bundle_data_dir", lambda: tmp_path / "bundle")

    dest = tmp_path / "appdata" / "glossaries"
    m._seed_user_glossaries(dest)

    assert (dest / "anime.json").is_file()
    assert (dest / "user.yaml").is_file()
    assert not (dest / "_seed.yaml").exists(), (
        "_seed.yaml must not be copied to app-data (copying freezes it at install-time version)"
    )
    assert not (dest / "_denylist.txt").exists()


def test_glossary_manager_creates_missing_dir_on_construction(tmp_path):
    """GlossaryManager must create its (user-writable) dir so a save can
    persist the genre JSON even when the dir doesn't exist yet -- e.g. the
    bundle seed copy failed or the source was missing. (_save_unlocked writes
    via tempfile.mkstemp(dir=...), which FileNotFoundErrors when the dir is
    absent.)

    Seeded defect: remove the mkdir in GlossaryManager.__init__ -> save()'s
    mkstemp raises FileNotFoundError -> this fails.
    """
    from yamabiko.romanization.glossary import GlossaryManager

    missing = tmp_path / "nope" / "glossaries"
    assert not missing.exists()
    gm = GlossaryManager(missing, genre="anime")
    gm.save()  # must not FileNotFoundError
    assert (missing / "anime.json").is_file()


def test_build_pipeline_honors_glossaries_dir_override_when_absolute(
    _pipeline_env, tmp_path,
):
    """Paired: an ABSOLUTE override path is trusted and passed through
    to GlossaryManager as-is, even if it doesn't exist. Users with
    custom glossaries in a known location want the override respected.

    Seeded-defect check: change the fallback condition to also
    override absolute paths (`if not gdir.exists():`) -> absolute
    override is dropped too, this assertion fails.
    """
    env = _pipeline_env
    custom_gdir = tmp_path / "my-custom-glossaries"
    custom_gdir.mkdir()
    _write_overrides(tmp_path, {
        "romanization": {
            "enabled": True,
            "glossaries_dir": str(custom_gdir),
        },
    })

    args = env.module.parse_args([])
    asyncio.run(env.module.build_pipeline(args))

    assert len(env.glossary_instances) == 1
    resolved_gdir = env.glossary_instances[0].gdir
    assert resolved_gdir == custom_gdir, (
        f"absolute glossaries_dir override was not honored; "
        f"expected {custom_gdir!r}, got {resolved_gdir!r}"
    )


def test_build_pipeline_falls_back_when_glossaries_dir_override_is_existing_relative(
    _pipeline_env, tmp_path, monkeypatch, caplog,
):
    """An EXISTING relative glossaries_dir override must STILL fall back
    to the bundled default (with a warning), because cwd differs between
    dev (repo root) and the frozen exe ($INSTDIR): the same relative
    string resolves to two different directories, so it can never be
    honored safely. Only absolute overrides are honored.

    Red today: the old guard was
        if not gdir.is_absolute() and not gdir.exists():
    so a relative path that HAPPENS to exist relative to cwd was honored
    verbatim -- the L-02 cwd-ambiguity bug. This test creates exactly
    that condition (chdir into tmp_path, create the relative dir there).

    Seeded-defect check: restore the ``and not gdir.exists()`` conjunct
    -> the existing relative dir is honored (resolved_gdir == the
    relative Path, not the default) -> red.
    """
    import logging as _logging

    from yamabiko import paths

    env = _pipeline_env
    # Make the relative override EXIST relative to cwd.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing_rel_glossaries").mkdir()
    _write_overrides(tmp_path, {
        "romanization": {
            "enabled": True,
            "glossaries_dir": "existing_rel_glossaries",  # relative + real
        },
    })

    args = env.module.parse_args([])
    with caplog.at_level(_logging.WARNING, logger="yamabiko"):
        asyncio.run(env.module.build_pipeline(args))

    assert len(env.glossary_instances) == 1
    resolved_gdir = env.glossary_instances[0].gdir
    expected_default = paths.app_data_dir() / "glossaries"
    assert resolved_gdir == expected_default, (
        f"an EXISTING relative glossaries_dir override was honored "
        f"instead of falling back: got {resolved_gdir!r}, expected the "
        f"bundled default {expected_default!r}. cwd differs between dev "
        f"and the installed app, so a relative path is ambiguous."
    )
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno >= _logging.WARNING]
    assert any("glossaries_dir" in m for m in msgs), (
        f"falling back on a relative override must warn naming the key; "
        f"got {msgs!r}"
    )


def test_build_pipeline_exits_43_with_dialog_when_config_load_fails(
    _pipeline_env, monkeypatch,
):
    """A RuntimeError from config loading (missing/empty bundled defaults
    -- a broken install) must take the same tailored-dialog + sys.exit(43)
    path as ASR-init and translator sanity-check failures. Without the
    guard it propagates as an unhandled exception, the process exits 1,
    and the watchdog shows its generic min-lifetime "failed to start"
    dialog with no pointer at the real cause.

    Seeded-defect check: remove the try/except around
    ``cfg = _load_config(args.config)`` in build_pipeline — the
    RuntimeError propagates instead of SystemExit(43), no dialog is
    recorded, both asserts go red.
    """
    env = _pipeline_env
    shown: list = []
    import yamabiko.ui.error_dialog as _ed
    monkeypatch.setattr(_ed, "show_startup_error",
                        lambda err: shown.append(err))

    def _broken_load(explicit):
        raise RuntimeError(
            "bundled defaults missing or empty: "
            "C:/fake/_internal/yamabiko/default_config.yaml"
        )
    monkeypatch.setattr(env.module, "_load_config", _broken_load)

    args = env.module.parse_args(["--max-segments", "0"])
    with pytest.raises(SystemExit) as ei:
        asyncio.run(env.module.build_pipeline(args))

    assert ei.value.code == 43, (
        f"expected tailored-error sentinel 43, got {ei.value.code!r}; "
        "the watchdog treats non-43 exits as generic crashes and "
        "restarts the pipeline up to 3x for a permanently broken install"
    )
    assert len(shown) == 1, (
        f"expected exactly one show_startup_error call; got {len(shown)}"
    )


def test_invalid_numeric_config_exits_43_before_resources_start(
    _pipeline_env, tmp_path, monkeypatch,
):
    env = _pipeline_env
    path = tmp_path / "invalid-numeric.yaml"
    path.write_text(
        "translation:\n  temperature: .nan\n",
        encoding="utf-8",
    )
    resource_calls = []
    monkeypatch.setattr(
        env.module,
        "build_asr",
        lambda *_args, **_kwargs: resource_calls.append("asr"),
    )
    import yamabiko.vad.silero_chunker as silero_mod
    monkeypatch.setattr(
        silero_mod,
        "SileroChunker",
        lambda *_args, **_kwargs: resource_calls.append("vad"),
    )
    shown = []
    monkeypatch.setattr(
        "yamabiko.ui.error_dialog.show_startup_error",
        lambda error: shown.append(error),
    )

    args = env.module.parse_args(["--config", str(path)])
    with pytest.raises(SystemExit) as exc:
        asyncio.run(env.module.build_pipeline(args))

    assert exc.value.code == 43
    assert resource_calls == []
    assert env.capture_instances == []
    assert env.translator_instances == []
    assert env.glossary_instances == []
    assert env.romanization_instances == []
    assert len(shown) == 1


@pytest.mark.asyncio
async def test_build_pipeline_exits_43_when_llama_never_ready(
    _pipeline_env, tmp_path, monkeypatch,
):
    """A readiness timeout is a TRUE startup failure -> exit 43 (the same
    tailored-error policy as a sanity_check failure). The watchdog must
    NOT restart the pipeline in a loop for a llama that can't load its GGUF
    (exit 43 suppresses the generic "keeps crashing" restart + dialog).

    Seeded defect: drop the `await translator.wait_until_ready(...)` call
    from build_pipeline -> no exit 43 on a never-ready llama -> this test
    stays in the `asyncio.run(build_pipeline(args))` call which returns a
    PipelineApp instead of raising SystemExit, causing the `pytest.raises`
    block to NOT catch SystemExit -> red.
    """
    from yamabiko import main as main_mod
    from yamabiko.translation.translator import LlamaNotReadyError

    class _NotReadyTranslator(_FakeTranslator):
        async def wait_until_ready(self, *, timeout_s, poll_s):
            raise LlamaNotReadyError(
                "llama-server did not become ready at http://127.0.0.1:8090/health"
                f" within {timeout_s:.0f}s"
            )

    monkeypatch.setattr(main_mod, "Translator", _NotReadyTranslator)

    from yamabiko.ui import error_dialog
    shown_errors: list = []
    monkeypatch.setattr(
        error_dialog, "show_startup_error",
        lambda e: shown_errors.append(e),
    )

    _write_overrides(tmp_path)
    args = main_mod.parse_args([])
    with pytest.raises(SystemExit) as ei:
        await main_mod.build_pipeline(args)

    assert ei.value.code == 43, (
        f"LlamaNotReadyError from wait_until_ready did not produce exit 43; "
        f"got {ei.value.code!r}. A cold-loading llama that never becomes ready "
        f"would restart the pipeline in a loop (watchdog sees non-43 exit = "
        f"generic crash -> 3x restarts -> generic give-up dialog)."
    )
    assert len(shown_errors) == 1, (
        f"expected exactly one show_startup_error call; got {len(shown_errors)}"
    )
    assert isinstance(shown_errors[0], error_dialog.StartupError), (
        f"show_startup_error received a {type(shown_errors[0]).__name__}, "
        f"not a StartupError; classify_startup_exception wiring broken"
    )
    assert shown_errors[0].title == "The translator didn't start", (
        f"LlamaNotReadyError must route to the llama_failed dialog (its message "
        f"names llama-server), not the generic fallback; got title "
        f"{shown_errors[0].title!r}"
    )


# ---------------------------------------------------------------------------
# Issue #54 recovery wiring: a capture stream death (device unplug / suspend /
# active-but-silent) must RESTART + re-resolve, not silently STOP. The detector
# lives in capture/audio.py; this is the main.py half -- the new exit code,
# the App flag + handler, the on_stream_death rewire, the pause probe, and the
# idle_death_s config consumer.
# ---------------------------------------------------------------------------

def test_build_pipeline_threads_idle_death_s_into_audio_config(_pipeline_env, tmp_path):
    """audio.idle_death_s must reach AudioConfig.idle_death_s (and thence the
    capture monitor's heartbeat window). A producer/consumer mismatch is the
    bug class hard rule #4 exists to catch.

    Seeded defect: drop idle_death_s from the AudioConfig(...) call in
    build_pipeline -> app.audio_cfg.idle_death_s falls back to the 90 default
    instead of the 137 override -> red. (Also needs the key in
    default_config.yaml, else the loader strips the override before it lands.)
    The AudioConfig.idle_death_s -> capture._idle_death_s copy is exercised
    separately by the test_audio.py idle-heartbeat tests (real LoopbackCapture;
    the env here wraps capture in a tracking double without that attribute).
    """
    env = _pipeline_env
    _write_overrides(tmp_path, {"audio": {"idle_death_s": 137}})
    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))
    assert app.audio_cfg.idle_death_s == 137, (
        f"audio.idle_death_s override did not reach AudioConfig "
        f"(got {app.audio_cfg.idle_death_s}); the heartbeat window is hardcoded"
    )


def test_build_pipeline_explicit_config_missing_idle_death_s_defaults_to_90(
    _pipeline_env, tmp_path,
):
    """An explicit --config is layered OVER the shipped defaults
    (user_overrides.yaml skipped), so a pre-release explicit config
    whose audio section predates the idle_death_s key must resolve the
    key from the defaults layer: build_pipeline succeeds, the MERGED
    cfg carries the shipped 90, and AudioConfig.idle_death_s lands on
    90. (Pre-merge history: the file was read verbatim and
    cfg["audio"]["idle_death_s"] direct indexing raised an uncaught
    KeyError -> watchdog fast-crash give-up; the defaults merge is now
    what restores the key, and the merged-cfg assert below pins THAT
    mechanism, not just the consumer's .get() fallback.)

    Seeded defect: skip the defaults merge in load_full_config's
    explicit branch (config.py: ``merged = _validate_and_merge(...)``
    -> ``merged = explicit``) -> app.cfg["audio"] has no idle_death_s
    key -> the merged-cfg assert goes red (KeyError). Drill run
    2026-07-11: applied, red; reverted, green.
    """
    import yaml as _yaml

    from yamabiko import paths as _paths

    env = _pipeline_env
    # Full explicit config built from the shipped schema, minus the
    # idle_death_s key, with the heavy collaborators disabled (matching
    # the fixture's stubs) -- the shape of a config file written by a
    # release that predates the key.
    cfg = _yaml.safe_load(
        _paths.shipped_defaults_path().read_text(encoding="utf-8")
    )
    del cfg["audio"]["idle_death_s"]
    cfg["romanization"]["enabled"] = False
    cfg["hotkeys"]["enabled"] = False
    cfg["overlay"]["browser"] = False
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(_yaml.safe_dump(cfg), encoding="utf-8")

    args = env.module.parse_args(["--config", str(legacy)])
    app = asyncio.run(env.module.build_pipeline(args))

    assert app.cfg["audio"]["idle_death_s"] == 90, (
        f"the defaults<-explicit merge must restore audio.idle_death_s "
        f"in the merged config; got "
        f"{app.cfg['audio'].get('idle_death_s', '<absent>')!r}"
    )
    assert app.audio_cfg.idle_death_s == 90, (
        f"audio section without idle_death_s must fall back to the shipped "
        f"default 90; got {app.audio_cfg.idle_death_s!r}"
    )


def test_request_stream_death_arms_flag_and_fires_done(_pipeline_env, tmp_path):
    """PipelineApp.request_stream_death (the body the capture monitor's
    on_stream_death callback ultimately runs) must arm stream_death_requested
    AND fire `done`, so amain returns EXIT_STREAM_DEATH (RESTART) and the
    pipeline tears down. Mirrors request_stop_all.

    Seeded defect: drop `self.done.set()` from request_stream_death -> the
    pipeline never unblocks on a capture death -> done assertion red. Drop the
    flag set -> stream_death_requested assertion red.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)
    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))
    assert app.stream_death_requested is False and not app.done.is_set()
    app.request_stream_death()
    assert app.stream_death_requested is True, "request_stream_death did not arm the flag"
    assert app.done.is_set(), "request_stream_death did not fire done"


def test_amain_returns_stream_death_restart_code(monkeypatch):
    """amain must return EXIT_STREAM_DEATH when the capture signalled a stream
    death (and no user quit). classify(44) -> RESTART, so the watchdog rebuilds
    and re-resolves the device. Without this, capture death returned 0
    (STOP_SELF) and the watchdog stopped -- the silent-stall's other half.

    Seeded defect: drop the stream_death_requested branch from amain's _code
    expression -> returns 0 -> red.
    """
    from yamabiko import main as main_mod
    from yamabiko.lifecycle.exit_codes import EXIT_STREAM_DEATH

    class _FakeApp:
        stop_all_requested = False
        stream_death_requested = True

    async def _fake_build(args):
        return _FakeApp()

    async def _fake_run(app):
        return None

    monkeypatch.setattr(main_mod, "build_pipeline", _fake_build)
    monkeypatch.setattr(main_mod, "run_pipeline", _fake_run)
    code = asyncio.run(main_mod.amain(main_mod.parse_args([])))
    assert code == EXIT_STREAM_DEATH, (
        f"amain returned {code} on a capture stream death; it must return "
        f"EXIT_STREAM_DEATH({EXIT_STREAM_DEATH}) so the watchdog RESTARTS"
    )


def test_amain_returns_stage_escalation_restart_code(monkeypatch):
    """amain must return EXIT_STAGE_ESCALATION when a stage crossed its
    supervisor escalation threshold (and no user quit / capture death).
    classify(45) -> RESTART, so the watchdog rebuilds the pipeline.
    Without this, escalation set `done` alone and amain returned 0
    (STOP_SELF): the watchdog logged "child exited cleanly" and cascaded
    a full stop -- the app silently died with no dialog and no restart.

    Seeded defect: drop the stage_escalation_requested branch from
    _exit_code_for -> amain returns 0 -> red.
    """
    from yamabiko import main as main_mod
    from yamabiko.lifecycle.exit_codes import EXIT_STAGE_ESCALATION

    class _FakeApp:
        stop_all_requested = False
        stream_death_requested = False
        stage_escalation_requested = True

    async def _fake_build(args):
        return _FakeApp()

    async def _fake_run(app):
        return None

    monkeypatch.setattr(main_mod, "build_pipeline", _fake_build)
    monkeypatch.setattr(main_mod, "run_pipeline", _fake_run)
    code = asyncio.run(main_mod.amain(main_mod.parse_args([])))
    assert code == EXIT_STAGE_ESCALATION, (
        f"amain returned {code} on a stage escalation; it must return "
        f"EXIT_STAGE_ESCALATION({EXIT_STAGE_ESCALATION}) so the watchdog "
        f"RESTARTS instead of reading a stage crash-loop as a clean quit"
    )


def test_amain_stop_all_wins_over_stream_death(monkeypatch):
    """If a user quit (stop_all) races a stream death on the same teardown,
    STOP_ALL(43) must win -- the watchdog stops, never restarts the app the
    user just asked to close.

    Seeded defect: reorder amain's _code ternary so stream_death is checked
    first -> a user quit racing a death returns 44 (RESTART) -> red.
    """
    from yamabiko import main as main_mod

    class _FakeApp:
        stop_all_requested = True
        stream_death_requested = True

    async def _fake_build(args):
        return _FakeApp()

    async def _fake_run(app):
        return None

    monkeypatch.setattr(main_mod, "build_pipeline", _fake_build)
    monkeypatch.setattr(main_mod, "run_pipeline", _fake_run)
    code = asyncio.run(main_mod.amain(main_mod.parse_args([])))
    assert code == 43, (
        f"a user quit racing a stream death returned {code}; STOP_ALL(43) must "
        f"win so the watchdog does not restart a deliberately-quit app"
    )


def test_run_pipeline_pause_probe_tracks_app_state(_pipeline_env, tmp_path):
    """The capture's idle-heartbeat pause probe must track the live
    app_state, so a deliberate pause suppresses the silent-stall heartbeat
    (a paused video legitimately goes to digital silence). Invokes the probe
    and asserts it tracks a real toggle (not just that it's set).

    Seeded defect: drop ``pause_state_probe=app_state.pause_snapshot`` from
    build_pipeline's LoopbackCapture(...) call -> the probe stays the default
    ``lambda: False`` and can't see a pause -> the paused assertion red.
    """
    env = _pipeline_env
    _write_overrides(tmp_path)
    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert app.capture._is_paused() is False
    app.app_state.pause()
    assert app.capture._is_paused() is True, (
        "the capture pause probe does not reflect a live pause; the heartbeat "
        "would fire on a deliberately paused (silent) source"
    )
    app.app_state.resume()
    assert app.capture._is_paused() is False


def test_run_pipeline_capture_death_callback_arms_restart(_pipeline_env, tmp_path):
    """Integration: the capture monitor fires on_stream_death from a foreign
    thread; run_pipeline must wire it so firing it marshals onto the loop, arms
    stream_death_requested, and fires done -> the pipeline exits for a watchdog
    RESTART. Fires the REAL wired callback from a REAL thread on a live loop
    (anti-pattern 9: not just asserts it's set), so both the cross-thread
    call_soon_threadsafe marshalling AND a lambda that forgot
    request_stream_death are exercised.

    Seeded defect: rewire on_stream_death back to `...call_soon_threadsafe(
    app.done.set)` (drops the flag) -> done still fires and the pipeline exits,
    but stream_death_requested stays False -> the flag assertion red.
    """
    import threading
    import time as _time

    env = _pipeline_env
    _write_overrides(tmp_path)

    async def _drive():
        args = env.module.parse_args([])
        app = await env.module.build_pipeline(args)

        def _fire_when_wired():
            # Fire from a REAL foreign thread (the capture monitor's thread in
            # production), so on_stream_death's call_soon_threadsafe genuinely
            # crosses the thread boundary instead of degenerating to call_soon
            # on the loop thread. on_stream_death is None at build; run_pipeline
            # wires it early (before the await-done point).
            for _ in range(2000):
                # getattr: the test's capture double has no on_stream_death attr
                # until run_pipeline wires it, and this real thread polls before
                # that (unlike the old on-loop task, which only ran after).
                cb = getattr(app.capture, "on_stream_death", None)
                if cb is not None:
                    cb()
                    return
                _time.sleep(0.005)

        t = threading.Thread(
            target=_fire_when_wired, name="test-fake-capture-monitor", daemon=True,
        )
        t.start()
        await asyncio.wait_for(env.module.run_pipeline(app), timeout=15.0)
        t.join(2.0)
        return app

    app = asyncio.run(_drive())
    assert app.stream_death_requested is True, (
        "firing the wired capture.on_stream_death did not arm "
        "stream_death_requested; the pipeline would exit 0 (STOP_SELF) and the "
        "watchdog would STOP instead of restarting + re-resolving the device"
    )
    assert app.done.is_set()


# ---------------------------------------------------------------------------
# Config identity line: digest + APPLIED override KEYS (logging-hardening
# Task 5). The process identity banner itself lives in main() (before the
# wizard gate) and is covered by tests/test_main_entry.py; build_pipeline
# emits the config fingerprint line, because only here does the MERGED
# config exist.
# ---------------------------------------------------------------------------


def test_build_pipeline_config_line_lists_applied_override_keys_not_raw_or_values(
    _pipeline_env, tmp_path, caplog,
):
    """build_pipeline must emit a ``config: digest=<12hex>
    overrides=[...]`` INFO line where the override-keys list reflects the
    keys load_full_config ACTUALLY applied -- keys only, never values
    (values hold endpoints/paths = PII in an attached bug-report log),
    and no keys the loader stripped. A typo'd key that unknown-key
    validation discarded must NOT be listed: a triager comparing the
    line against behavior would otherwise chase an override that had no
    effect on the run.

    Seeded defects: (a) delete the config-line log call in
    build_pipeline -> no ``config: `` record, first assert red; (b) list
    keys from the RAW user_overrides.yaml instead of the applied set ->
    the stripped typo key appears, its absence assert red; (c) emit
    key=value pairs -> the value-absence assert red.
    """
    import logging
    import re

    env = _pipeline_env
    _write_overrides(tmp_path, {
        "genre": "anime",
        # Unknown to the shipped schema: load_full_config warns and
        # strips it, so it is never applied to the merged config.
        "definitely_not_a_real_key": "sekrit-value",
    })

    args = env.module.parse_args([])
    with caplog.at_level(logging.INFO, logger="yamabiko"):
        asyncio.run(env.module.build_pipeline(args))

    config_lines = [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("config: ")
    ]
    assert config_lines, (
        "build_pipeline logged no 'config: ...' fingerprint line"
    )
    line = config_lines[0]
    assert re.search(r"digest=[0-9a-f]{12}\b", line), (
        f"no 12-hex config digest on the config line: {line!r}"
    )
    assert "genre" in line, (
        f"applied override KEY 'genre' missing from the config line: {line!r}"
    )
    assert "definitely_not_a_real_key" not in line, (
        f"a typo'd key the loader STRIPPED is listed as an applied "
        f"override (raw-file read instead of applied-key set): {line!r}"
    )
    assert "anime" not in line and "sekrit-value" not in line, (
        f"override VALUE leaked into the config line (PII class): {line!r}"
    )


def test_build_pipeline_config_line_reports_source_file_for_explicit_config(
    _pipeline_env, tmp_path, caplog,
):
    """Under ``--config <file>`` the explicit file is layered over the
    shipped defaults with the user_overrides.yaml layer SKIPPED, so
    there are no USER overrides to list: the config line must
    fingerprint the explicit file and name it as
    ``source=--config:<basename>`` with NO override-keys list. The
    pre-fix behavior hashed the explicit file while listing keys from
    user_overrides.yaml -- a digest whose key list belongs to a file the
    run never applied.

    Seeded defects: (a) keep reading override keys from
    user_overrides.yaml in --config mode -> 'genre'/'overrides='
    appears, red; (b) drop the source naming -> the source assert red.
    """
    import logging
    import re

    import yaml as _yaml

    from yamabiko import paths as _paths

    env = _pipeline_env
    # A user_overrides.yaml EXISTS alongside (the trap: its keys must
    # not leak into the --config run's fingerprint line).
    _write_overrides(tmp_path, {"genre": "anime"})

    # Full verbatim config from the shipped schema with the heavy
    # collaborators disabled (matching the fixture's stubs) -- same
    # pattern as the explicit-config threshold test above.
    cfg = _yaml.safe_load(
        _paths.shipped_defaults_path().read_text(encoding="utf-8")
    )
    cfg["romanization"]["enabled"] = False
    cfg["hotkeys"]["enabled"] = False
    cfg["overlay"]["browser"] = False
    explicit = tmp_path / "explicit-run.yaml"
    explicit.write_text(_yaml.safe_dump(cfg), encoding="utf-8")

    args = env.module.parse_args(["--config", str(explicit)])
    with caplog.at_level(logging.INFO, logger="yamabiko"):
        asyncio.run(env.module.build_pipeline(args))

    config_lines = [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("config: ")
    ]
    assert config_lines, (
        "build_pipeline logged no 'config: ...' fingerprint line in "
        "--config mode"
    )
    line = config_lines[0]
    assert re.search(r"digest=[0-9a-f]{12}\b", line), (
        f"no 12-hex config digest on the config line: {line!r}"
    )
    assert "source=--config:explicit-run.yaml" in line, (
        f"--config mode must name the explicit file as the digest "
        f"source: {line!r}"
    )
    assert "overrides=" not in line and "genre" not in line, (
        f"--config mode applies no user overrides (user_overrides.yaml "
        f"is skipped), yet the config line carries override keys -- and "
        f"they came from a file this run never read: {line!r}"
    )


def test_build_pipeline_emits_junk_logging_level_warning_itself(
    _pipeline_env, tmp_path, caplog,
):
    """build_pipeline must consume the junk logging.level warning that
    load_full_config STASHES (config.consume_level_warning) and emit it
    from its own ``yamabiko`` logger. In production, main() attaches the
    main-pipeline.log handler BEFORE build_pipeline runs, so this
    sequenced re-emission is what puts the evidence in the pipeline
    child's role file -- the loader itself no longer emits (an in-load
    emission is pre-handler-attach in the WATCHDOG process and dies on
    the frozen bundle's discarded stderr, so emission is every role's
    own job).

    The record is asserted on logger name ``yamabiko`` (build_pipeline's
    ``log``), not ``yamabiko.config``: a loader-side emission must not
    satisfy this test.

    Seeded defects (each one line):
      - drop the consume_level_warning() + log.warning block in
        build_pipeline -> no ``yamabiko``-logger warning names the junk
        value -> red.
      - drop the _normalize_merged_log_level stash in load_full_config
        -> consume returns None, nothing emitted -> red.
    """
    import logging

    env = _pipeline_env
    _write_overrides(tmp_path, {"logging": {"level": "verbose"}})
    args = env.module.parse_args([])
    with caplog.at_level(logging.WARNING, logger="yamabiko"):
        app = asyncio.run(env.module.build_pipeline(args))

    assert app.cfg["logging"]["level"] == "INFO", (
        f"junk logging.level did not normalize to INFO in the merged "
        f"config; got {app.cfg['logging']['level']!r}"
    )
    own_warnings = [
        r.getMessage() for r in caplog.records
        if r.name == "yamabiko" and r.levelno == logging.WARNING
        and "logging.level='verbose'" in r.getMessage()
    ]
    assert own_warnings, (
        "build_pipeline did not re-emit the stashed junk-level warning "
        "from its own logger; on the frozen path the pipeline child's "
        "main-pipeline.log carries no trace of the rejected value"
    )


# ---------------------------------------------------------------------------
# Translator health-flip logging (logging-hardening Task 9)
# ---------------------------------------------------------------------------


def test_note_translator_health_warns_once_per_unhealthy_transition(
    _pipeline_env, tmp_path, caplog,
):
    """PipelineApp.note_translator_health must log ONE warning on the
    healthy->unhealthy flip -- not one per 10 s poll (a down llama would
    otherwise write ~6 warnings/min for its whole outage) -- and re-arm
    after a recovery so the NEXT outage warns again. It must also keep
    feeding the /healthz cache on every call.

    Seeded defect: drop the previous-state guard (log the warning
    whenever reachable is False) -> three unhealthy polls produce three
    warnings, the ==1 assert goes red.
    """
    import logging

    env = _pipeline_env
    _write_overrides(tmp_path)
    args = env.module.parse_args([])
    app = asyncio.run(env.module.build_pipeline(args))

    with caplog.at_level(logging.INFO, logger="yamabiko"):
        # Three consecutive failing polls: exactly one warning.
        app.note_translator_health(False, RuntimeError("connect refused"))
        app.note_translator_health(False, RuntimeError("connect refused"))
        app.note_translator_health(False, RuntimeError("connect refused"))
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "translator health" in r.getMessage()
        ]
        assert len(warnings) == 1, (
            f"expected exactly 1 warning for 3 consecutive failing polls; "
            f"got {len(warnings)}: {[r.getMessage() for r in warnings]!r}"
        )
        assert app._translator_reachable_cached is False, (
            "/healthz cache not updated on the failing poll"
        )

        # Recovery: cache flips back, and the guard re-arms.
        app.note_translator_health(True)
        assert app._translator_reachable_cached is True, (
            "/healthz cache not updated on recovery"
        )
        app.note_translator_health(False, RuntimeError("connect refused"))
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "translator health" in r.getMessage()
        ]
        assert len(warnings) == 2, (
            "the second outage (after a recovery) did not warn again; "
            "the transition guard never re-armed"
        )


class _HealthClient:
    """The health refresher's client during a problem: answers /health."""

    def __init__(self, status: int):
        self.status = status
        self.urls: list[str] = []

    async def get(self, url, *a, **kw):
        self.urls.append(url)
        response = _FakeHttpResponse()
        response.status_code = self.status
        return response


def test_translator_problem_follows_the_health_probe(_pipeline_env, tmp_path):
    """Lead decision 2 / Decisions 19 and 71: once /v1/models has answered,
    a failing probe raises the translator problem once; /v1/models 200 while
    /health says 503 (still loading) keeps it; /health 200 clears it.

    Seeded defects: raise on every failing tick, or clear on /v1/models
    alone -> red."""
    env = _pipeline_env
    _write_overrides(tmp_path)
    app = asyncio.run(env.module.build_pipeline(env.module.parse_args([])))
    app.manager = _FakeFastAPIManager()
    app.translator.readiness_url = "http://127.0.0.1:0/health"

    async def _drive():
        await app.track_translator_problem(True, _HealthClient(200))
        await app.track_translator_problem(False, _HealthClient(503))
        await app.track_translator_problem(False, _HealthClient(503))
        loading = _HealthClient(503)
        await app.track_translator_problem(True, loading)
        assert loading.urls == ["http://127.0.0.1:0/health"]
        assert app.app_state.problem_kinds() == ("translator",)
        await app.track_translator_problem(True, _HealthClient(200))

    asyncio.run(_drive())
    frames = [f for f in app.manager.broadcasts if f["type"] == "problems"]
    assert [[p["kind"] for p in f["active"]] for f in frames] == [["translator"], []]


def test_a_completed_translation_clears_the_translator_problem(_pipeline_env, tmp_path):
    """Decision 19: a translation that completed proves the translator
    answers. Seeded defect: never clear from the stage -> red."""
    from yamabiko import problems

    env = _pipeline_env
    _write_overrides(tmp_path)
    app = asyncio.run(env.module.build_pipeline(env.module.parse_args([])))
    app.manager = _FakeFastAPIManager()
    asyncio.run(app.note_translation_ok())
    assert app.manager.broadcasts == []
    app.app_state.set_problem(problems.translator_problem(False))
    asyncio.run(app.note_translation_ok())
    assert app.app_state.problem_kinds() == ()
    assert app.manager.broadcasts[-1] == {"type": "problems", "active": [], "recovered": []}


def test_a_failing_probe_raises_nothing_when_a_translation_completed_since_the_last_tick(
    _pipeline_env, tmp_path,
):
    """Decision 19 (review M6): once /v1/models has answered (Decision 71),
    a failing probe right after a completed translation (a probe that timed
    out at 2 s) is not a translator problem; the next failing tick with no
    translation in between raises it.

    Seeded defect: raise from the probe regardless of translations -> red."""
    env = _pipeline_env
    _write_overrides(tmp_path)
    app = asyncio.run(env.module.build_pipeline(env.module.parse_args([])))
    app.manager = _FakeFastAPIManager()

    async def _drive():
        await app.track_translator_problem(True, _HealthClient(200))
        await app.note_translation_ok()
        await app.track_translator_problem(False, _HealthClient(503))
        assert app.app_state.problem_kinds() == ()
        await app.track_translator_problem(False, _HealthClient(503))
        assert app.app_state.problem_kinds() == ("translator",)

    asyncio.run(_drive())


def test_a_failing_probe_raises_nothing_while_v1_models_never_answered(_pipeline_env, tmp_path):
    """Decision 71 (review I2): an endpoint that never served /v1/models (a
    remote server without the route) gets no "Translations stopped" from the
    probe, at launch or after a quiet stretch; the translation escalations
    cover it.

    Seeded defect: raise from a failing tick before /v1/models answered
    -> red."""
    env = _pipeline_env
    _write_overrides(tmp_path)
    app = asyncio.run(env.module.build_pipeline(env.module.parse_args([])))
    app.manager = _FakeFastAPIManager()

    async def _drive():
        await app.track_translator_problem(False, _HealthClient(404))
        await app.track_translator_problem(False, _HealthClient(404))

    asyncio.run(_drive())
    assert app.manager.broadcasts == []


def test_translator_copy_promises_a_restart_only_under_the_watchdog_with_the_bundled_server(
    _pipeline_env, tmp_path, monkeypatch,
):
    """Lead decision 4: "Yamabiko is restarting it" only when the watchdog
    supervises the bundled loopback llama-server. Seeded defects: always
    promise it, or ignore a remote endpoint -> red."""
    env = _pipeline_env
    _write_overrides(tmp_path)
    args = env.module.parse_args([])
    assert asyncio.run(env.module.build_pipeline(args)).translator_restarts is False
    monkeypatch.setenv("YAMABIKO_SHUTDOWN_EVENT_PIPELINE", "Local\\yamabiko.test")
    assert asyncio.run(env.module.build_pipeline(args)).translator_restarts is True
    _write_overrides(tmp_path, {"translation": {
        "endpoint": "http://192.168.1.20:8080/v1/chat/completions"}})
    assert asyncio.run(env.module.build_pipeline(args)).translator_restarts is False


def test_the_probe_promises_a_restart_only_when_the_server_is_down_and_an_escalation_upgrades_it(
    _pipeline_env, tmp_path, monkeypatch,
):
    """Lead decision 4 (final review M1): under the watchdog with the bundled
    server, a failing probe that connected and got no answer (hung; nothing
    restarts it yet) raises the copy without "Yamabiko is restarting it"; one
    that could not connect (crashed; its supervisor respawns it) raises the
    restart copy. The translation escalation, which restarts the hung
    server, upgrades the entry in place (Decision 43: a content change, no
    second notification) and still holds the exit.

    Seeded defects: promise the restart from every probe under the
    watchdog, or keep the escalation from upgrading the entry -> red."""
    import httpx

    from yamabiko import problems

    env = _pipeline_env
    _write_overrides(tmp_path)
    monkeypatch.setenv("YAMABIKO_SHUTDOWN_EVENT_PIPELINE", "Local\\yamabiko.test")

    async def _failing_tick_after_models_answered(exc):
        app = await env.module.build_pipeline(env.module.parse_args([]))
        app.manager = _FakeFastAPIManager()
        await app.track_translator_problem(True, _HealthClient(200))
        app.note_translator_health(False, exc)
        await app.track_translator_problem(False, _HealthClient(503))
        return app

    hung = asyncio.run(_failing_tick_after_models_answered(httpx.ReadTimeout("read timed out")))
    assert [f["active"] for f in hung.manager.broadcasts] == [[problems.translator_problem(False)]]
    crashed = asyncio.run(_failing_tick_after_models_answered(httpx.ConnectError("refused")))
    assert [f["active"] for f in crashed.manager.broadcasts] == [[problems.translator_problem(True)]]

    holds: list[float] = []

    async def _hold(delay, result=None):
        holds.append(delay)
        return result

    hung.manager.clients = 1
    monkeypatch.setattr(asyncio, "sleep", _hold)
    hung.on_stage_escalate("translation")
    asyncio.run(hung.announce_restart_problems())
    assert [f["active"] for f in hung.manager.broadcasts] == [
        [problems.translator_problem(False)], [problems.translator_problem(True)]]
    assert holds == [env.module._ANNOUNCE_HOLD_S]


def test_translation_escalation_announces_the_problem_and_carries_a_translator_restart(
    _pipeline_env, tmp_path, monkeypatch,
):
    """Lead decisions 2 and 4, Decision 18: a translation escalation
    broadcasts the translator problem before the exit and leaves a carry that
    asks the watchdog to restart llama-server; a user quit leaves none.

    Seeded defects: skip the broadcast, skip the carry, or carry on a quit
    -> red."""
    from yamabiko import problems

    env = _pipeline_env
    _write_overrides(tmp_path)
    monkeypatch.setenv("YAMABIKO_SHUTDOWN_EVENT_PIPELINE", "Local\\yamabiko.test")
    app = asyncio.run(env.module.build_pipeline(env.module.parse_args([])))
    app.manager = _FakeFastAPIManager()
    app.on_stage_escalate("translation")
    asyncio.run(app.announce_restart_problems())
    assert app.manager.broadcasts[-1]["active"] == [problems.translator_problem(True)]
    assert problems.take_carry() == problems.Carry(("translator",), True)

    quit_app = asyncio.run(env.module.build_pipeline(env.module.parse_args([])))
    quit_app.manager = _FakeFastAPIManager()
    quit_app.on_stage_escalate("translation")
    quit_app.request_stop_all()
    asyncio.run(quit_app.announce_restart_problems())
    assert quit_app.manager.broadcasts == []
    assert problems.take_carry() is None


def test_a_restart_announcement_keeps_the_translator_problem_through_the_hold(
    _pipeline_env, tmp_path, monkeypatch,
):
    """Decision 70 (review I1): once a restart exit is announced, neither a
    /health 200 tick nor a completed translation during the hold lifts the
    translator problem, so "Translations are back." never shows before the
    restart; the next pipeline reports the recovery through the carry.

    Seeded defect: lift during the announcement -> red."""
    from yamabiko import problems

    env = _pipeline_env
    _write_overrides(tmp_path)
    app = asyncio.run(env.module.build_pipeline(env.module.parse_args([])))
    app.manager = _FakeFastAPIManager()
    app.manager.clients = 1
    holds: list[float] = []

    async def _probe_and_translate_during_the_hold(delay, result=None):
        holds.append(delay)
        await app.track_translator_problem(True, _HealthClient(200))
        await app.note_translation_ok()
        return result

    monkeypatch.setattr(asyncio, "sleep", _probe_and_translate_during_the_hold)
    app.on_stage_escalate("translation")
    asyncio.run(app.announce_restart_problems())
    assert holds == [env.module._ANNOUNCE_HOLD_S]
    assert app.app_state.problem_kinds() == ("translator",)
    assert [f["active"] for f in app.manager.broadcasts] == [[problems.translator_problem(False)]]


def test_a_translation_escalation_restarts_llama_server_only_when_it_is_not_restarting_itself(
    _pipeline_env, tmp_path, monkeypatch,
):
    """Decision 21 (review I3): llama-server whose last health probe could
    not connect (it crashed) or answered /health 503 (its supervisor's
    respawn is loading the model) is already restarting, so the carry keeps
    the translator problem but asks for no second restart; one that
    accepted the probe and never answered (hung) is restarted.

    Seeded defects: restart llama-server for every translation escalation,
    or skip it for every probe failure (a hung server stays hung) -> red."""
    import httpx

    from yamabiko import problems

    env = _pipeline_env
    _write_overrides(tmp_path)
    monkeypatch.setenv("YAMABIKO_SHUTDOWN_EVENT_PIPELINE", "Local\\yamabiko.test")

    async def _escalate_after(probe):
        app = await env.module.build_pipeline(env.module.parse_args([]))
        app.manager = _FakeFastAPIManager()
        await probe(app)
        app.on_stage_escalate("translation")
        await app.announce_restart_problems()
        assert app.manager.broadcasts[-1]["active"] == [problems.translator_problem(True)]
        return problems.take_carry()

    async def _crashed(app):
        app.note_translator_health(False, httpx.ConnectTimeout("connect timed out"))

    async def _loading(app):
        await app.track_translator_problem(True, _HealthClient(503))

    async def _hung(app):
        app.note_translator_health(False, httpx.ReadTimeout("read timed out"))

    assert asyncio.run(_escalate_after(_crashed)) == problems.Carry(("translator",), False)
    assert asyncio.run(_escalate_after(_loading)) == problems.Carry(("translator",), False)
    assert asyncio.run(_escalate_after(_hung)) == problems.Carry(("translator",), True)


def test_stream_death_announces_the_audio_problem_only_when_the_endpoint_is_gone(
    _pipeline_env, tmp_path, monkeypatch,
):
    """Lead decision 2, R1: a capture death whose endpoint is still an active
    render endpoint restarts silently; one whose endpoint is gone announces
    the audio problem, carries it, and holds the exit _ANNOUNCE_HOLD_S while
    a client is connected, so the notification and the line are on screen
    before teardown stops the overlay.

    Seeded defects: announce every stream death, or never; skip the hold,
    or hold when nothing was raised -> red."""
    from yamabiko import problems

    env = _pipeline_env
    _write_overrides(tmp_path)
    app = asyncio.run(env.module.build_pipeline(env.module.parse_args([])))
    app.manager = _FakeFastAPIManager()
    app.manager.clients = 1
    holds: list[float] = []

    async def _record_hold(delay, result=None):
        holds.append(delay)
        return result

    monkeypatch.setattr(asyncio, "sleep", _record_hold)
    app.capture.endpoint_name = lambda: "Headphones (USB Audio)"
    monkeypatch.setattr(env.module, "active_render_endpoint_names",
                        lambda: ["Speakers (Realtek(R) Audio)", "Headphones (USB Audio)"])
    app.request_stream_death()
    asyncio.run(app.announce_restart_problems())
    assert app.manager.broadcasts == [] and holds == []
    assert problems.take_carry() is None
    monkeypatch.setattr(env.module, "active_render_endpoint_names",
                        lambda: ["Speakers (Realtek(R) Audio)"])
    asyncio.run(app.announce_restart_problems())
    assert app.manager.broadcasts[-1]["active"] == [problems.audio_problem("Headphones (USB Audio)")]
    assert problems.take_carry() == problems.Carry(("audio",), False)
    assert holds == [env.module._ANNOUNCE_HOLD_S]


def test_carried_problems_become_recoveries_for_new_clients(_pipeline_env, tmp_path, monkeypatch):
    """Decision 18: the next pipeline turns YAMABIKO_CARRIED_PROBLEMS into
    recovered entries of the connect frame. Seeded defect: never read the
    variable -> red."""
    import time as _time

    env = _pipeline_env
    _write_overrides(tmp_path)
    monkeypatch.setenv("YAMABIKO_CARRIED_PROBLEMS", "audio")
    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))
    assert app.app_state.problems_frame(now=_time.monotonic())["recovered"] == [
        {"kind": "audio", "recovery": "Listening again."}]


def test_run_pipeline_hands_each_health_probe_to_the_translator_problem(
    _pipeline_env, tmp_path, monkeypatch,
):
    """Wiring (TESTING.md 3.9, Decision 19): run_pipeline's health refresher
    passes every probe result to track_translator_problem. Here the
    translator problem is active and the fake client answers /v1/models and
    /health 200, so the first tick clears it.

    Seeded defects: drop the track_translator_problem call, or hand it
    models_ok=False -> red."""
    from yamabiko import problems

    env = _pipeline_env
    _write_overrides(tmp_path)
    monkeypatch.setattr(_FakeHttpResponse, "status_code", 200)

    async def _drive():
        app = await env.module.build_pipeline(env.module.parse_args([]))
        app.app_state.set_problem(problems.translator_problem(False))
        app.done.set()
        await env.module.run_pipeline(app)
        return app

    assert asyncio.run(_drive()).app_state.problem_kinds() == ()


def test_run_pipeline_announces_a_restart_problem_before_it_exits(
    _pipeline_env, tmp_path, monkeypatch,
):
    """Wiring (TESTING.md 3.9, Decision 18): once the wait ends, run_pipeline
    calls announce_restart_problems, so a stream death whose endpoint is
    gone leaves the carry for the watchdog. /v1/models answers 200 here, so
    the refresher adds no translator problem to the carry.

    Seeded defect: drop the announce_restart_problems call -> red."""
    from yamabiko import problems

    env = _pipeline_env
    _write_overrides(tmp_path)
    monkeypatch.setattr(_FakeHttpResponse, "status_code", 200)
    monkeypatch.setattr(env.module, "active_render_endpoint_names",
                        lambda: ["Speakers (Realtek(R) Audio)"])

    async def _drive():
        app = await env.module.build_pipeline(env.module.parse_args([]))
        app.capture.endpoint_name = lambda: "Headphones (USB Audio)"
        app.request_stream_death()
        await env.module.run_pipeline(app)

    asyncio.run(_drive())
    assert problems.take_carry() == problems.Carry(("audio",), False)
