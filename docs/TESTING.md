# Testing charter for Yamabiko

**Read this before editing any file under `tests/`.**

This document is binding. It exists so the test suite measures whether
the code works, not merely whether it runs.

---

## 0. Why this document exists

A test suite can show hundreds of passing tests while the shipping
build is broken in ways users see immediately. That gap is the reason
this document is here.

The failure mode is rarely "we need more tests." It is:

> **A test suite built to confirm assumptions rather than challenge
> them. A test that fails to catch a bug usually does so because it was
> constructed so the bug's failure mode was unreachable by design.**

A test like that ships green over a real defect because its structure
made the defect invisible. Those structures fall into a small number of
recognizable patterns, catalogued in §3 below, each paired with a
concrete example from this repo and the structural property that lets a
bug through.

This document exists so you cannot write one of those tests again
without knowing you are doing it.

### 0.1 Testing supports delivery

The purpose of this charter is to reject weak tests, not to maximize the
number of tests or the time spent designing them. The requested behavior is
the deliverable. Tests are one verification tool.

- If a plan or the user names an exact test scope, it is required and caps
  optional coverage. It cannot remove a mandatory requirement (§4.2 real-path
  coverage, §3.4 frozen-bundle coverage, the release drill). Add tests outside
  it only for a concrete new user-visible defect or with user approval; record
  other findings as follow-ups (§9).
- Test behaviors and regressions, not every helper, method, branch, schema, or
  dataclass introduced while implementing them.
- Research, benchmark, evidence-collection, and diagnostic code under
  `scripts/` may use a documented real-path probe (command, inputs, observed
  output) instead of automated tests. The exemption stops at `yamabiko/`:
  production code added or changed to serve such a tool follows the normal
  rules, and a script that becomes a release gate or ships loses the
  exemption.
- Do not create fixtures, parser matrices, mutation campaigns, generalized
  test infrastructure, or tests of tests unless that work is explicitly in
  scope.
- For ordinary feature work, establish one useful red result within 15
  minutes, then implement. Test authoring should stay below 25% of the task's
  elapsed work unless tests are themselves the requested deliverable.
- Sixty minutes without a completed production artifact or plan item is a
  mandatory stop-and-rescope checkpoint: stop and tell the user what blocked
  progress.

---

## 1. The Iron Law

> A test has value if and only if **it can fail when the code is
> broken in the way the test's name claims to verify.**

Operationally, this decomposes to three questions to answer before you
commit any test. Questions 1 and 3 must be answered yes; question 2 must be
answered yes or met by its stated alternative.

1. **Can I describe a one-line change to the production code that
   would make this test fail?** If no, the test does not verify what
   its name claims. Rewrite.

2. **For a new behavior or reproduced regression, did I observe the
   expected red result before implementing?** If no, confirm that an
   existing sensitive test already covered the behavior or explain why
   a manual real-path probe is the better verification shape. Green on
   first write is a reason to inspect the path, not an automatic demand
   for more test machinery.

3. **If I hand this test to a colleague and ask them to inject a
   defect this test should catch, can they do it in under 30
   seconds?** If injecting the defect takes longer than writing the
   fix, the test is testing ghosts.

Martin Fowler: *"If my tests never failed, either my programming is
perfect or my tests are inadequate. I know which way I'd bet."*

### Coverage is not sensitivity

A test suite can cover most lines and still catch less than half of
single-line defects. Line coverage says "the line executed"; mutation
testing says "if the line were wrong, you would notice." For wiring
and composition code those are nearly independent questions: it is
routine to see a composition module at high line coverage with a
mutation kill rate under half (see §6.2).

Every decision in this document optimizes for sensitivity, not
coverage. When they conflict, sensitivity wins.

---

## 2. The pre-merge checklist

Before you commit **any** test, confirm each applicable item below. The
checklist is conditional: `N/A` is valid and non-blocking when the test does
not involve that risk. Do not manufacture work merely to make every row apply.

- [ ] **Defect-sensitivity check.** I can describe a simple production
      defect that would make this test fail. Actually seed the defect when
      the test is unexpectedly green, the regression is high-risk, or the
      release checklist requires the drill.

- [ ] **No subject-mock in a real test.** If this test is at the
      `@pytest.mark.real` tier, it does not use `MagicMock` / `Mock`
      / `AsyncMock` for the *thing being verified*. Mocking is
      allowed *around* the subject (fixtures, teardown, unrelated
      collaborators) but never *as* the subject.

- [ ] **Name matches body.** If the name says "spawns X" / "handles
      Y" / "survives Z" / "round-trips through Q", the body actually
      causes X, triggers Y, monitors Z over time, or exercises
      producer to consumer on Q. If I wrote the body first, I renamed
      to match what the body actually does.

- [ ] **Observable effects, not mock call counts.** Assertions
      target disk state, network response, process exit code,
      rendered pixels, log contents, or exception types: something
      the user or OS could observe. Not `mock.call_count == 1`. Not
      `mock.called_with(X)`. Those are permitted only as *additional*
      assertions after an observable-effect one.

- [ ] **For startup / E2E / supervised-process tests: survival, not
      snapshot.** If the subject under test is a process that a
      supervisor may respawn, the test samples PIDs at two time points
      separated by at least 5 seconds and asserts PID stability
      between them. A single snapshot cannot distinguish "healthy"
      from "crash-loop."

- [ ] **For startup / E2E tests: log asserted clean.** The test scans
      the relevant log's new content during its execution window and
      fails on any of `Traceback`, `AttributeError`, `ImportError`,
      `ModuleNotFoundError`, `OSError`, or `ERROR pipeline stage`. A
      test that writes to a log but never reads it is log-blind and
      hides every uncaught exception.

- [ ] **For composition / wiring code: the test invokes the entry
      point.** If the bug class is "wiring between components", the
      test calls `main()` / `build()` / `run()` of the relevant script
      with the blocking event loop mocked to return 0 or with the test
      running in a worker thread. Construction of individual components
      is not enough; the most damaging wiring bugs hide in the
      entry-point body, not in the constructors.

- [ ] **For callbacks / handlers / drivers this change adds or whose body
      it changes: a test invokes it directly.** If `build_X` attaches a callback to
      `app.hotkey_mgr.register(on_pause=_on_pause)`, stores a signal
      handler on `app.screen_change_handler`, or registers an
      on-escalate callback, there must be a paired test that *invokes*
      the callable and asserts on the side effect (state changed,
      broadcast fired, event set). Wiring-only tests are §3.9
      Wired-not-exercised and cannot distinguish a correctly-registered
      callable from a no-op lambda.

- [ ] **For shipping external-dependency call sites this change adds or
      modifies (HF, pyaudio, ONNX, subprocess, tokenizers, keyboard,
      httpx, PySide6): real-path coverage exists.** "Modifies" includes
      changed arguments, input shapes or dtypes, and a changed dependency
      version or model pin. "Shipping" means bundle code plus release-gate
      scripts (`scripts/preflight.py`, `scripts/run_pytest_release_phase.py`).
      The covering `@pytest.mark.real` test exercises the real library
      (network, OS, GPU, display), not a mock of it. The enumeration is
      open-ended: *any* integration seam qualifies, not just the ones
      listed here.

**If every applicable box is checked, commit. If any applicable box is
unchecked, fix or drop the test.** After one focused red result, write the
production code. Do not batch additional tests first, and do not reopen a
completed task for speculative coverage findings.

---

## 3. The nine anti-patterns

Every pattern below is named, defined, given a concrete example from
this codebase, and given the pattern the fixed test uses. When you
review a test (yours or someone else's), check each of these.

### 3.1 The Liar

**Definition.** A test whose assertion is so weak it is evergreen:
satisfied regardless of whether the subject works.

**Example.** An overlay launch regression once crashed `main()` on
every start with `AttributeError: 'Signal' object has no attribute
'connect'`. A test named for spawning the pipeline and overlay
asserted `len(yamabiko_children) >= 2` at a **single snapshot at
t=3s**. The watchdog respawned the crashing overlay, so two yamabiko
children were always alive at any instant. The test passed against a
catastrophically broken build.

**Correct pattern.** The assertion must be false when the subject is
broken. For supervised processes, sample PIDs at t=3s and again at
t=10s and assert the overlay PID is identical at both (a crash-loop
shows a different PID at the second sample). For startup, assert no
`Traceback` in the log's test window.

**Sub-taxonomy.** The Liar is four distinct failure modes that need
different fixes. When you find a Liar, name which flavor it is:

1. **Structural Liar.** Asserts on attribute presence / type / id
   without observing behavior. `assert app.mgr is not None` when the
   fixture pre-supplies a non-None fake. Fix: assert against a
   fixture-owned sentinel identity, or assert isinstance of the
   concrete type.

2. **Behavioral Liar.** Subject is wired correctly and the test
   verifies the wiring, but the subject's body is never *invoked*:
   hotkey closures, `run_X` driver functions, signal handlers. Fix:
   invoke the subject and assert on side effects (state changes,
   broadcasts sent, events set). See §3.9; this is a named
   anti-pattern in its own right.

3. **Noise Liar.** Mutation is a string literal, log message, or
   dict-key rename (`"type"` to `"XXtypeXX"`). A test would only catch
   it by asserting on implementation-detail strings, which ages
   badly. **Do not chase.** A large fraction of mutation survivors are
   this flavor. Accept it as unavoidable noise.

4. **Equivalent-mutation Liar.** Mutation genuinely has no observable
   effect (for example `if x >= 0` versus `if x > 0` when `x` is always
   positive in practice). Unkillable by definition. Ignore.

Only (1) and (2) are worth tightening against. The rule: when a
survivor is pointed out, classify before chasing.

**See also.** Yegor Bugayenko's [Unit Testing Anti-Patterns](https://www.yegor256.com/2018/12/11/unit-testing-anti-patterns.html)
(Liar); Codepipes [Software Testing Anti-patterns](https://blog.codepipes.com/testing/software-testing-antipatterns.html) (The Liar).

### 3.2 Mocked-the-bug-away

**Definition.** The test mocks the exact call site where the bug
lives. The mock, being a `MagicMock`, accepts anything and confirms
nothing.

**Example.** A real-tier HF test originally used `MagicMock` to stand
in for `huggingface_hub.hf_hub_download`. The real production call
passed `tqdm_class=...`, which the real library rejects with
`TypeError`. The mock accepted the kwarg silently. Production crashed
on the first download; the test was green.

**Correct pattern.** At the `@pytest.mark.real` tier the subject must
be the real library, with no mocks on the subject. The fixed test,
`test_all_wizard_hf_repos_exist_with_expected_files`, calls
`huggingface_hub.list_repo_files()` against the actual HF API. If the
library signature changes or a repo returns 401, the test fails.

**See also.** Khorikov, *Unit Testing Principles, Practices, and
Patterns*, ch. on "Mocks and Test Fragility." Enterprise
Craftsmanship's [Structural Inspection anti-pattern](https://enterprisecraftsmanship.com/posts/structural-inspection/).

### 3.3 Construction-not-invocation

**Definition.** The test constructs an object but never calls the path
where the bug lives, typically `main()`, `build()`, or an
initialization sequence.

**Example.** A `_QGuiApp.primaryScreenChanged.connect(...)` call lived
inside `overlay_app/__main__.py:main()`. No test invoked `main()`;
every overlay test constructed `OverlayWindow`, `Tray`,
`PipelineClient`, and `SettingsWindow` directly, bypassing the `main()`
body. The faulty line was unreachable by the suite, so a crash on every
launch shipped green. Separately, the overlay and the bundled
`llama-server` were never launched by any test, because all tests used
`--run-main` (which dispatches straight to the pipeline) and skipped
the top-level watchdog spawn logic where those launches are triggered.

**Correct pattern.** For every shipping entry point
(`yamabiko/main.py`, watchdog, `overlay_app/__main__.py`, wizard) and
every release-gate script (`scripts/preflight.py`,
`scripts/run_pytest_release_phase.py`) there must be at least one test
that invokes it end-to-end with the blocking step (`app.exec()`, the
pipeline loop) mocked to return. Research and diagnostic scripts follow
§0.1. The template is
`tests/overlay_app/test_main_signal_connects.py::test_overlay_main_reaches_completion_without_attribute_error`.

**See also.** Feathers, *Working Effectively with Legacy Code* (the
Humble Object pattern). Seemann, [Composition Root](https://blog.ploeh.dk/2011/07/28/CompositionRoot/).
Fowler, [HumbleObject](https://martinfowler.com/bliki/HumbleObject.html).

### 3.4 Dev-venv-only

**Definition.** The test runs against the developer's Python
environment, where every dependency is installed natively. The frozen
PyInstaller bundle is a fundamentally different world: different import
semantics, different resource-path resolution, different DLL search
paths, different stdout behavior.

**Examples.** Several regressions passed every dev-venv test and
crashed immediately in the frozen install:

- `hf_xet` absent from the spec, capping single-stream downloads.
- `cublas64_12.dll` missing from the bundle (in the dev venv, torch's
  `lib/` had supplied it).
- `unidic_lite` PEP 420 namespace-package shadowing from a missing
  `include_py_files=True` on `collect_data_files`.
- `cutlet/exceptions.tsv` not collected as a data file.
- `print(flush=True)` raising `OSError [Errno 22]` in a windowed child
  spawned with only `stdin` redirected (its stdout is a broken pipe).
- `hf_hub_download` failing before its first byte with "'NoneType' object
  has no attribute 'write'" in a pipeline started without std handles:
  tqdm's progress bar writes to `sys.stderr`, which was `None`. The
  default-launch tests could not see it, because the watchdog hands its
  children broken pipes, not `None`.

**Correct pattern.** Any test whose failure mode depends on
packaging / import / path / env must run against the frozen bundle,
not the dev venv. `tests/test_real_frozen_bundle.py` and
`tests/test_real_frozen_e2e.py` are the home. Critical: an "imports key
modules" test must actually invoke the import chain that breaks, not
just run `--version`, which short-circuits before the heavy imports.

**See also.** PyInstaller's [Common Issues and Pitfalls](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html).

### 3.5 Structural inspection

**Definition.** The test asserts on the shape of internal state
(attribute names present, dict keys exist, classes instantiated)
instead of observable external behavior.

**Example.** The `StartupError` factory tests in
`tests/test_error_dialog.py` construct a `StartupError` dataclass and
assert its fields. For a long time the function that actually renders
the error to the user, `show_startup_error()`, was never called, so a
production change that broke the dialog rendering would have passed
unchanged. The factory tests are now paired with
`tests/overlay_app/test_error_window.py::test_show_startup_error_shows_the_window_with_the_tail_ending_on_its_dialog_line`,
which drives `show_startup_error()` to the real window and asserts what
the user sees (the title, a report fact, the log tail ending on the
dialog line, the saved theme, the placement), with the real-tier tests in
`tests/overlay_app/test_real_error_window.py`, and with the frozen
`tests/test_real_frozen_e2e.py::test_frozen_startup_failure_shows_the_report_window_and_copy_report_survives_the_exit`,
which seeds Models missing in the bundle (a speech model set by hand
that isn't downloaded, started offline) and copies the report through
UI Automation.

**Correct pattern.** For UI components, trigger a render and assert on
observable output (window created, text visible, exit code on
dismissal). For data pipelines, run the pipeline and assert on its
output, not on which classes were constructed.

**See also.** Khorikov, *Unit Testing Principles, Practices, and
Patterns*, ch. 5. [Enterprise Craftsmanship: Structural Inspection](https://enterprisecraftsmanship.com/posts/structural-inspection/).

### 3.6 Single-snapshot (crash-loop-masking)

**Definition.** A one-time observation of process or resource state
that cannot distinguish a healthy process from a crash-then-respawn
cycle. Sibling of The Liar.

**Correct pattern.** For any test involving supervised processes,
sample state at t=N and again at t=N+Δ (Δ at least 5s), then assert
continuity (PID stable, log marker absent, no new child processes
created). For any assertion of the form "X should exist," follow it
with "X should still exist at t+Δ with the same identity."

### 3.7 Log-blind

**Definition.** The test exercises code that writes to a log but never
reads the log. Uncaught exceptions, silent fallbacks, and
crash-respawn cycles all write to logs; a test that does not read the
log cannot see any of them.

**Examples.** An `OSError [Errno 22]` from `print()` under the windowed
bootloader produced a traceback in the pipeline log, but no test read
that log. A signal-wiring crash produced an `AttributeError` traceback
on every respawn, with no test reading it.

**Correct pattern.** Every E2E and real-path test that writes to a log
ends by scanning the log's new content since test start. Logs are
per-process: the watchdog writes `main-watchdog.log`, the
pipeline / wizard child writes `main-pipeline.log`, and the overlay
child writes `main-overlay.log`, all under
`%LOCALAPPDATA%\Yamabiko\logs\`. The frozen E2E suite reads them with
its `_LogTail` helper (a merged tail over the `main-*.log` glob) and
`_wait_for_log_marker`. Any line containing `Traceback`,
`AttributeError`, `ImportError`, `ModuleNotFoundError`, `OSError`, or
`ERROR pipeline stage` fails the test. This is a small, copy-pasteable
addition to any startup test.

### 3.8 Name-body drift

**Definition.** The test's name makes a strong claim; the body
verifies a much weaker property. The gap between name and body is
where bugs live, because reviewers read the name, trust the claim, and
do not verify the body.

**Examples.**

- A "frozen bundle imports key modules" test whose body ran
  `yamabiko.exe --version`, which short-circuits *before* importing
  `yamabiko.main`, the overlay, the audio capture, or any of the
  actually-important modules. The current suite splits this into
  `test_frozen_bundle_version_prints_without_import_error` (the cheap
  `--version` short-circuit) and
  `test_frozen_bundle_imports_pipeline_module_tree_without_exceptions`
  (which forces the real import chain).
- A "watchdog starts subprocess" test whose body injects a
  `_FakeProcess` factory, so no OS subprocess is started. The name is
  true in the Watchdog-class sense but not the OS sense.
- A "spawns pipeline and overlay" test whose body asserted
  `len(children) >= 2` at one instant; it actually verified "two
  yamabiko processes exist right now," a different and weaker claim.

**Correct pattern.** Apply the rename-or-match rule: for every
strong-claim name, either strengthen the body to match or rename to
the weaker, accurate claim.

- "starts X" → body spawns X, or rename to `test_X_policy_*`.
- "handles X" → body triggers X and observes the response.
- "survives N seconds" → body monitors for N seconds with stability
  assertions.
- "works" / "is correct" / "is valid" / "is safe" → **banned**;
  rename to something specific.

**See also.** Adzic, [*Specification by Example*](https://www.manning.com/books/specification-by-example).

### 3.9 Wired-not-exercised

**Definition.** A callable, closure, or handler is *constructed* and
*registered* (passed as a callback, attached to a signal, stored on an
App dataclass) but its *body* is never invoked by any test. Wiring is
verified; behavior is not. It is one level deeper than §3.3:
§3.3 is "build components, never call `main()`"; this one is "call
`main()`, wire the components, never invoke the components'
callbacks."

**Example.** The `_on_pause` and `_on_toggle_overlay` hotkey closures
inside `run_pipeline` were tested for wiring (`app.hotkey_mgr is
env.fake_hotkey_mgr`, `register()` called with `callable()` callbacks)
but never invoked. Mutation testing independently surfaced three
survivors at that seam:

- `if app.manager is not None` → `is None` (invert the guard)
- `snap = app.app_state.snapshot()` → `snap = None`
- `"type": "state"` → `"XXtypeXX": "state"` (dict key rename)

Each is a real production bug: a hotkey press silently no-ops, crashes
with `AttributeError`, or broadcasts the wrong payload shape. All would
have passed the suite as written. The same pattern applies to every
`run_X` driver function that pairs with a `build_X` factory:
`run_watchdog` once hit a 0% mutation kill rate because no test invoked
it at all. A `run_X` driver you add or change needs a direct invocation
test, not just a "build returns the right dataclass" test.

**Correct pattern.** For every callable / closure / handler you add or change, a
paired test **invokes** it (typically via `asyncio.run(method())` for
async, a direct call for sync) and asserts on the side effect that
invocation produces: state change, broadcast, event set, method
recorded. If the wiring lives inside a local closure in `run_X`,
extract it onto the App dataclass as a method (as
`PipelineApp.on_stage_escalate` and `handle_pause_hotkey` were); the
method is trivial to test directly, and the original closure becomes a
thin thread-bridge adapter.

**See also.** §6.2 for how to run mutation testing to find more of
these.

---

## 4. Positive requirements by tier

The two-tier system (unit and `@pytest.mark.real`) is correct. What
matters is what each tier permits.

### 4.1 Unit tier (default `pytest`)

- **Speed:** a full unit run completes in seconds.
- **Scope:** a single unit of behavior: one function, one class, one
  pure-data pipeline step.
- **Mocks:** allowed and encouraged for speed and isolation, with one
  constraint: if the test name contains "spawn", "launch", "connect",
  "download", "load", "render", or "show", the body must exercise the
  real path or be renamed.
- **Coverage of bug classes:** the unit tier cannot catch
  dev-vs-frozen, subprocess-tree, GUI-render, or OS-integration bugs.
  Do not try to make it.
- **Name must match body** at the unit tier as strictly as at any
  other tier.

### 4.2 Real-path tier (`@pytest.mark.real`)

- **Speed:** individual tests may take seconds; the overall suite may
  take minutes. Acceptable.
- **Scope:** the seam between Yamabiko and an external dependency:
  `huggingface_hub`, `pyaudiowpatch`, `onnxruntime`,
  `subprocess.Popen`, `tokenizers`, `keyboard`, `httpx`, `PySide6`,
  the filesystem, the CUDA runtime. Any library that can change under
  us or behave differently in prod versus dev is a real-path-tier
  subject.
- **Mocks on the subject: banned.** If the test name says `test_real_X`
  or the marker says `@pytest.mark.real`, the subject of the test must
  be the real thing. A real-tier test that mocks its subject is a
  mislabelled unit test; move it or rewrite it.
- **Observable effects required.** Disk state, network response, exit
  code, rendered pixel, process tree, log contents. Not
  `mock.assert_called_with(...)`.
- **Every shipping external-dependency call site this change adds or
  modifies** needs a real-path test on that call site. "Modifies" includes
  changed arguments, input shapes or dtypes, and a changed dependency version
  or model pin. "Shipping" means bundle code plus release-gate scripts
  (`scripts/preflight.py`, `scripts/run_pytest_release_phase.py`). The
  enumeration above is open, not closed: any integration seam qualifies,
  including PySide6, which the signal-wiring crash later proved.

### 4.3 Frozen-bundle E2E (`tests/test_real_frozen_*.py`)

- **Subject:** the actual `dist/yamabiko/yamabiko.exe`, spawned from
  the actual install tree.
- **Requirements:** apply every §2 row the subject involves; for frozen
  E2E, "survival, not snapshot" and "log asserted clean" are never `N/A`.
- **Positive sentinels:** assert ready-markers are logged
  (`pipeline.ready`, `overlay.mainwindow.shown`). Not asserting
  sentinels means "nothing visibly broke in N seconds," which is
  weaker than "the thing that should happen did happen."
- **Skips:** allowed when a resource is unavailable (no bundle, no
  GPU, no audio) via an explicit `pytest.skip("reason")`. Never
  silently via a conditional that returns `True`.

### 4.4 Installer-level smoke (CI, `windows-latest`)

- **Subject:** the real NSIS installer, built and run on a fresh
  GitHub runner. Defined in `.github/workflows/installer-smoke.yml`,
  driven by `installer/ci-smoke.ps1`.
- **Steps:** silent install (`Setup.exe /S /D=...`), launch
  `yamabiko.exe --version` and `--help` from the installed location,
  then silent uninstall via `Uninstall.exe /S` with cleanup
  verification (user data and HF cache survive; install files are
  removed). Runtime logs are uploaded on failure.
- **Pages are not covered here.** The smoke is silent, so the installer's
  pages, the Still running box raised from Welcome, the Finish page's
  Start Yamabiko and the Removed page's size are checked by the
  "Installer pages" step in `docs/RELEASE.md`; `tests/test_installer_nsi.py`
  pins their static contracts. The script saves and restores the shared
  uninstall registry key and the Start Menu's Yamabiko folder, so a local
  run leaves an installed Yamabiko as it was.
- **Llama is stubbed in CI.** Every installer this workflow builds uses
  stubbed llama binaries, so these artifacts can install, run
  `--version`, and uninstall but cannot translate. Tests that need a
  live translator (the full `pipeline.ready` sentinel, the
  sanity-check path) run locally against a real build, not in this
  workflow.
- **Triggers:** this heavyweight workflow runs on the `main` path
  (`dev` to `main` PRs, pushes to `main`, `v*` tags) and on a weekly
  schedule, not on every PR.

---

## 5. Stack-specific gotchas

These are traps specific to this stack. Internalize them.

### 5.1 PySide6 signals

- `SomeQClass.mySignal` returns a `Signal` **descriptor** (no
  `.connect`). Signals are bound only on **instances**.
  `instance.mySignal.connect(...)` works; `Class.mySignal.connect(...)`
  raises `AttributeError`. The overlay launch crash happened because
  nobody tested `main()` end-to-end; a one-line invocation would have
  surfaced it in milliseconds.

- `QApplication.exec()` blocks. To unit-test `main()`:
  `monkeypatch.setattr(QApplication, "exec", lambda self: 0)`.
  Combined with mocking `PipelineClient.start` / `stop` and skipping
  any update-check thread, `main()` returns 0 quickly and every
  signal-wiring line is exercised.

- `QApplication.instance() or QApplication([])`: pytest-qt fixtures
  pre-create the app, so do not try to create a second one (it raises
  `RuntimeError`). When testing `main()`, do not use the `qapp`
  fixture; let `main()` create the app itself.

- `setHighDpiScaleFactorRoundingPolicy` must run **before**
  `QApplication(...)`. In tests this emits a `QtWarningMsg` if the
  fixture beat you to it; benign, can be ignored.

- A typed `Signal(str, str, list)` **marshals an emitted `None` to
  `[]`** (each slot arg is coerced to the declared C++ type) before any
  slot sees it. Consequence for mutation testing: an `or []`
  null-coercion immediately ahead of `.emit()` is an *equivalent
  mutation* (§3.1.4), unkillable at that seam, so do not chase it.

### 5.2 PyInstaller (dev vs frozen)

- `Path(__file__).parent` points at the source file in dev and at
  `_internal/...` in the frozen bundle. Use `paths.bundle_data_dir()`
  or check `getattr(sys, 'frozen', False)` → `sys._MEIPASS`.

- `cwd` in dev is typically the repo root. In the frozen install, cwd
  is the install directory, so relative paths in `default_config.yaml`
  resolve differently. Rule: no relative paths in defaults; use
  absolute or computed paths.

- `collect_data_files("package")` grabs the data tree but not the `.py`
  source. If the package's `__init__.py` holds constants (such as
  `unidic_lite`'s `DICDIR`), you need `include_py_files=True`, or
  PyInstaller creates a PEP-420 namespace package on disk that shadows
  the PYZ-embedded one.

- `collect_data_files("package")`, even with `include_py_files=True`,
  can miss data files the package reads via
  `Path(__file__).parent / 'foo.tsv'`. Those need explicit inclusion
  (for example `cutlet/exceptions.tsv`).

- Native package DLLs need `collect_dynamic_libs`, which preserves the
  wheel's layout (`_internal/nvidia/cublas/bin/cublas64_12.dll`). The
  PyInstaller bootloader adds `_internal/` (root) to the DLL search
  path but **not** its subdirectories. If the loading code calls
  `LoadLibrary("cublas64_12.dll")` without a full path, you must
  flatten the DLLs into the `_internal/` root.

- A windowed `yamabiko.exe` started without std handles (Explorer, the
  Start Menu, a `DETACHED_PROCESS` spawn) has `sys.stdout` and
  `sys.stderr` set to `None`, and libraries write to `sys.stderr` anyway
  (tqdm inside `hf_hub_download`). A child it spawns with only `stdin`
  redirected gets broken pipes instead, where `print(..., flush=True)`
  raises `OSError [Errno 22]`. The frozen entry (`yamabiko/watchdog.py`,
  `_guard_frozen_stdio`) gives a `None` stream an `os.devnull` sink and
  wraps both in `_SafeTextIO` after the module's first standard-library
  imports, before the Win32 bindings and any third-party or Yamabiko
  import. Only a frozen test that starts the role exactly that way can see
  either class.

### 5.3 Subprocess supervision

- Mocking `subprocess.Popen` tests the policy engine (restart logic,
  flap detection, exit-code handling). It does NOT test the seam
  (command construction, flag correctness, OS-specific behavior). Both
  need coverage, each in its own tier.

- A single process-tree snapshot of a supervised crash-loop looks
  identical to a healthy run. Always sample twice with PID-stability
  assertions (§3.6).

- Windows `CREATE_NEW_PROCESS_GROUP` has side effects on signal
  delivery: you can `os.kill(pid, signal.CTRL_BREAK_EVENT)` to the
  group, but `CTRL_C_EVENT` is ignored by the group leader. Test the
  real flag, not a mocked `Popen`.

### 5.4 Qt + asyncio

- Do NOT run pytest-asyncio and pytest-qt in the same test body against
  the same app instance without `qasync.QEventLoop` or similar; they
  fight over the loop. Isolate: use `qtbot.waitUntil(...)` for
  Qt-driven flows, or `asyncio.run(...)` for async-only flows.

- For pipeline tests, `asyncio.run(...)` is the default. Only
  overlay / Qt tests need `qtbot`.

---

## 6. Verifying tests can fail

Two complementary tools keep the Iron Law honest: a manual drill for
confidence in a specific fix, and mutation testing for an exhaustive
audit of a composition refactor.

### 6.1 The seeded-defect drill (pre-release)

The drill operationalizes Fowler's principle: *"If my tests never
failed, either my programming is perfect or my tests are inadequate."*
Before any `v*` tag, this is a required step in `docs/RELEASE.md`:

1. On a throwaway branch, pick one known-bug fix from the rotation
   list below.
2. Revert that commit (`git revert <sha>` or a manual reversion).
3. Before building, run `scripts/preflight.py` and
   `pytest -m "real and not frozen_artifact"`.
4. Build the exact candidate, then run all post-build gates:
   - `pytest tests/test_real_frozen_bundle.py -m "real and not frozen_artifact" -v`
   - `pytest tests/test_real_frozen_bundle.py -m frozen_artifact -v --junitxml=build/frozen-artifact-gate.xml`, with zero `<skipped>` test cases in the JUnit report
   - `scripts/preflight.py --release`
   - the rotation row's named/relevant frozen E2E test
   - the installer smoke workflow
5. **Expected: the rotation row's named detector is red.** If that detector
   stays green, the test suite is lying. Do **not** tag the release. Fix the
   named test first, then repeat the drill.

**Rotation list.** The rotation table (which commit to revert and which
test must go red) lives in `docs/RELEASE.md` under "Seeded-defect drill".
That table is the single source; pick one row per release and rotate
through it. If a row's test stays green under revert, that test has
drifted; it is the first thing to fix.

### 6.2 Mutation testing with mutmut (ad-hoc, not a CI gate)

The seeded-defect drill tests one defect at a time. Mutation testing
(mutmut) tests every single-line defect mutmut can generate, thousands
per file, and reports which ones the test suite fails to catch.

**When to run.** Only when a plan, audit, or release task schedules it;
schedule one after a significant composition refactor. When a run is
scheduled, read the baseline first and append the new kill rates
afterward. **Not** on every PR: a full run on the composition modules
takes hours even scoped. **Not** on utility modules where the kill rate
is naturally high (pure computation, dataclasses); the signal adds
little there.

**How to run.** The config lives in `setup.cfg`. On Windows, set two
env tweaks to work around mutmut 2.x encoding bugs (mutmut 3.x requires
WSL; 2.x works natively with these):

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
.venv\Scripts\python.exe -m mutmut run
```

Then classify survivors with `scripts/mutmut-analyze.py`; it buckets
them by composition function and computes the per-function kill rate.

**How to interpret survivors.** Classify each survivor into one of the
four Liar flavors (§3.1):

- **Structural** or **Behavioral** Liar → real gap. Tighten the test.
  This is the signal mutmut is for.
- **Noise** (log strings, dict-key renames, error-message text) →
  leave alone. Tightening means asserting on implementation details
  that age badly.
- **Equivalent mutation** (no observable behavior change) → ignore.

**Kill-rate expectations.**

| Surface | Typical kill rate | Interpretation |
|---|--:|---|
| Composition / wiring | 45-55% | Expected after serious test work; residual survivors are mostly noise |
| Data-transform code | 70-90% | Pure computation; most mutations produce observable wrong output |
| Config merge / validation | 60-80% | If you have tier coverage |
| Pipeline stage loops | 10-20% | Hard to test in isolation; behavioral coverage requires a real pipeline |
| UI rendering | 10-30% | Observable behavior lives in pixels, which unit tests cannot see |

**Do not** set a hard kill-rate gate. Around half on composition is
fine; 20% on a stage loop is fine; 90% on a four-field dataclass is
also fine. Context rules.

**What mutmut cannot do.**

- Catch bugs that require multiple simultaneous mutations (a symptom
  that depends on two wrong branches taken together).
- Catch temporal / concurrency bugs.
- Tell you WHICH survivors are noise versus real gaps; that requires
  human classification (§3.1 sub-taxonomy).

**Artifacts.** `setup.cfg` holds the mutmut config.
`scripts/mutmut-analyze.py` is the per-function classifier. The delta
between runs is the real signal, not the absolute numbers.

---

## 7. Naming conventions

Test names are executable specifications (Adzic). Use strong, specific
verbs:

- `test_X_does_Y`: body exercises the path where X does Y.
- `test_X_handles_Z_by_W`: body triggers Z and observes W.
- `test_X_survives_N_seconds_under_Y`: body monitors over N seconds
  with stability assertions under load Y.
- `test_X_round_trips_through_Y`: body writes via the producer, reads
  via the consumer, asserts equality.
- `test_X_fails_fast_on_Y`: body injects Y and asserts timely failure.

**Banned vague verbs** in test names (reviewers reject on sight):

- "works"
- "is valid"
- "is correct"
- "is safe"
- "is right"

If you cannot name the specific observable behavior the test verifies,
you do not know what the test verifies. Rename before committing.

---

## 8. Anti-cheat: things that make a test green but useless

These patterns pass the checklist above *textually* while violating it
structurally. Don't.

- **`try: ... except Exception: pass`** inside a test body. Any broad
  exception swallow during test execution is a sign of a test that
  does not want to see failures. A `finally` branch for cleanup is
  fine; silencing assertion-adjacent exceptions is not.

- **`assert True`** or **`assert 1 == 1`** in any form, including
  `assert x or True` and `assert not (x and False)`.

- **`assert len(x) >= 0`** and siblings (`x is not None` on a
  freshly-constructed literal, `isinstance(x, dict)` on a dict
  literal). These are tautologies.

- **Conditional skips based on the thing being tested**, for example
  `if not hasattr(app, 'signal'): pytest.skip(...)`. That is exactly
  the bug the test should catch, routed around.

- **"Smoke" tests that only smoke-test the test.** `import X` is not a
  test of X. `X()` is also not a test of X unless the constructor has
  observable side effects worth asserting.

- **Snapshot tests without a diff check.** Approval tests are fine;
  "capture the output, trust it forever" tests are not.

---

## 9. The review stance

Review a test once against its named claim and the risks that actually apply.
The default is skepticism, not unlimited expansion. A concrete defect outside
the task's agreed test scope becomes a follow-up unless it invalidates the
requested behavior. Before you approve:

1. Read the name. Form an expectation of what a defect would look
   like.
2. Read the body. Imagine the defect. Would the body catch it?
3. If you are not sure, identify the exact production defect first. Run a
   seeded-defect check only when the uncertainty is material.
4. Look for each of §3's anti-patterns. Name them explicitly in review
   comments if present ("this is 3.2 Mocked-the-bug-away").
5. Stop after the claim is demonstrated. Do not turn review into a new test
   design task.

---

## 10. The literature

These are the primary sources. Read the first three if you read
nothing else.

- **Fowler (2021):** [*On the Diverse and Fantastical Shapes of
  Testing*](https://martinfowler.com/articles/2021-test-shapes.html).
  Plus the line: *"If my tests never failed, either my programming is
  perfect or my tests are inadequate. I know which way I'd bet."*
- **Feathers (2004):** *Working Effectively with Legacy Code*:
  characterization tests, Humble Object.
- **Khorikov (2020):** *Unit Testing Principles, Practices, and
  Patterns*: Structural Inspection, mocks versus stubs.
- **Seemann (2011):** [Composition
  Root](https://blog.ploeh.dk/2011/07/28/CompositionRoot/). Applied to
  Qt apps: refactor `main()` to `build(argv) → App; app.run()`.
- **Adzic (2011):** *Specification by Example*. Rename-or-match.
- **Dodds:** [*Write tests. Not too many. Mostly
  integration.*](https://kentcdodds.com/blog/write-tests)
- **Yegor Bugayenko:** [Unit Testing
  Anti-Patterns](https://www.yegor256.com/2018/12/11/unit-testing-anti-patterns.html).
- **Codepipes:** [Software Testing
  Anti-patterns](https://blog.codepipes.com/testing/software-testing-antipatterns.html).

---

## 11. When this document is wrong

This document is not sacred. If you find a pattern that shipped a bug
that is not catalogued here, add it. If you find a rule that is causing
net harm (slowing you more than it saves), raise it; do not silently
work around it.

The minimum bar is that the rule had a chance to fire before you
override it. Every rule here has a concrete defect it was written in
response to. That is the bar for adding new rules too.

Process budgets (time limits, scope ceilings) cite the plan or project rule
that motivated them.
