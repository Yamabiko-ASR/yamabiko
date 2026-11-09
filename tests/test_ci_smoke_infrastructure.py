"""Meta-tests for the installer-smoke CI infrastructure.

The workflow at .github/workflows/installer-smoke.yml + the two
PowerShell helpers under installer/ci-*.ps1 are test infrastructure,
not product code. They nonetheless have contracts the rest of the
project relies on:

* ci-prepare-vendor-stubs.ps1 must stub `llama-server.exe` (the one
  file installer/build.ps1 throws on if missing) so the CI build
  completes without the real vendored binary.
* ci-smoke.ps1 must exercise both --version AND --help against the
  INSTALLED exe (the historical gap that dist-only smoke missed).
* installer-smoke.yml must declare the heavyweight ``installer-smoke``
  job and trigger on pull_request + push to main. Commit e9da89c
  (2026-04-24) split the fast ``unit-tests`` + ``real-tier-tests`` jobs
  out into ``ci.yml`` (which runs on every dev + main PR/push); this
  file now guards BOTH workflows so the split can't silently drop a job
  or a trigger branch.

These tests fail when someone edits the CI config in a way that
silently drops coverage -- e.g., removing the --help smoke from
ci-smoke.ps1 would still pass every CI run but no longer detect a
broken help text. The same silent-pass guard that applies to unit tests
applies here.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STUBS_SCRIPT = _REPO_ROOT / "installer" / "ci-prepare-vendor-stubs.ps1"
_SMOKE_SCRIPT = _REPO_ROOT / "installer" / "ci-smoke.ps1"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "installer-smoke.yml"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _pytest_commands(step: dict) -> list[str]:
    """Return actual pytest command lines from one workflow step."""
    return [
        line.strip()
        for line in str(step.get("run", "")).splitlines()
        if re.match(r"pytest\b", line.strip())
    ]


def _marker_expression(command: str) -> str | None:
    """Extract pytest's ``-m`` expression from a workflow command."""
    match = re.search(
        r"(?:^|\s)-m\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))",
        command,
    )
    if match is None:
        return None
    return next(group for group in match.groups() if group is not None)


def _selected_marker_probe_tests(tmp_path: Path, expression: str) -> set[str]:
    """Ask pytest which sentinel tests a workflow marker expression selects."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\n"
        "markers =\n"
        "    real: external-dependency test\n"
        "    frozen_artifact: requires a built frozen candidate\n",
        encoding="utf-8",
    )
    probe = tmp_path / "test_marker_probe.py"
    probe.write_text(
        "import pytest\n\n"
        "def test_unmarked_unit():\n"
        "    pass\n\n"
        "@pytest.mark.real\n"
        "def test_lightweight_real():\n"
        "    pass\n\n"
        "@pytest.mark.real\n"
        "@pytest.mark.frozen_artifact\n"
        "def test_frozen_artifact():\n"
        "    pass\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--strict-markers",
            "-m",
            expression,
            str(probe),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"pytest rejected workflow marker expression {expression!r}:\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return {
        line.rsplit("::", 1)[-1]
        for line in proc.stdout.splitlines()
        if "::test_" in line
    }


def _workflow_code_lines(step: dict) -> list[str]:
    """Return executable, non-comment lines from a workflow ``run`` step."""
    return [
        line.strip()
        for line in str(step.get("run", "")).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


_BUILD_SCRIPT_INVOCATION = re.compile(
    r"(?:"
    r"&\s+[\"'](?:\.[\\/])?installer[\\/]build\.ps1[\"']"
    r"|(?:&\s+)?(?:\.[\\/])?installer[\\/]build\.ps1"
    r")",
    flags=re.IGNORECASE,
)
_ARTIFACT_PYTEST_INVOCATION = re.compile(
    r"pytest\s+tests[\\/]test_real_frozen_bundle\.py\s+"
    r"-m\s+(?:frozen_artifact|\"frozen_artifact\"|'frozen_artifact')\s+"
    r"-v\s+--timeout(?:=|\s+)60\s+"
    r"--junitxml(?:=|\s+)"
    r"(?:build[\\/]frozen-artifact-gate\.xml|"
    r"\"build[\\/]frozen-artifact-gate\.xml\"|"
    r"'build[\\/]frozen-artifact-gate\.xml')",
    flags=re.IGNORECASE,
)
_PYTEST_EXIT_GUARD = re.compile(
    r"if\s*\(\s*\$LASTEXITCODE\s+-ne\s+0\s*\)\s*"
    r"\{\s*exit\s+\$LASTEXITCODE\s*\}",
    flags=re.IGNORECASE,
)


def _command_junit_path(command: str) -> str | None:
    match = re.search(
        r"(?:^|\s)--junitxml(?:=|\s+)(?:\"([^\"]+)\"|'([^']+)'|(\S+))",
        command,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return next(group for group in match.groups() if group is not None)


def _normalized_workflow_path(path: str) -> str:
    return path.replace("\\", "/").casefold()


def _assert_installer_artifact_gate_contract(
    installer: dict,
    tmp_path: Path,
) -> tuple[str, str]:
    """Prove the built-candidate route and return its live skip-audit block."""
    assert installer.get("defaults", {}).get("run", {}).get("shell") == "pwsh", (
        "workflow defaults.run.shell must be exactly pwsh so causal run "
        "steps cannot be wrapped by an inert custom shell template"
    )

    job = installer["jobs"]["installer-smoke"]
    assert "if" not in job, (
        "the installer-smoke job must be unconditional and may not declare "
        "a disabling or dynamic if condition"
    )
    assert "shell" not in job.get("defaults", {}).get("run", {}), (
        "installer-smoke must inherit the top-level pwsh shell and may not "
        "override defaults.run.shell"
    )
    steps = job["steps"]
    build_routes = [
        (index, step)
        for index, step in enumerate(steps)
        if isinstance(step, dict)
        and len(_workflow_code_lines(step)) == 1
        and _BUILD_SCRIPT_INVOCATION.fullmatch(_workflow_code_lines(step)[0])
    ]
    assert len(build_routes) == 1, (
        "installer-smoke must contain exactly one actual installer/build.ps1 "
        "invocation; a display name or string mention is not a build boundary; "
        f"found {[index for index, _ in build_routes]}"
    )
    build_index, build_step = build_routes[0]
    assert "shell" not in build_step, (
        "installer/build.ps1 must inherit pwsh; shell override forbidden"
    )
    assert "if" not in build_step, (
        "the actual installer/build.ps1 step must be unconditional and may "
        "not declare a disabling or dynamic if condition"
    )
    assert build_step.get("continue-on-error", False) is False, (
        "the actual installer/build.ps1 step must not use true or dynamic "
        "continue-on-error; a failed build must block the job"
    )

    artifact_routes: list[tuple[int, dict, str]] = []
    for index, step in enumerate(steps):
        if index <= build_index or not isinstance(step, dict):
            continue
        for command in _pytest_commands(step):
            expression = _marker_expression(command)
            if expression is None:
                continue
            selected = _selected_marker_probe_tests(
                tmp_path / f"postbuild-{index}", expression
            )
            if selected == {"test_frozen_artifact"}:
                artifact_routes.append((index, step, command))

    assert len(artifact_routes) == 1, (
        "installer-smoke must have exactly one pytest selector after the "
        "actual installer/build.ps1 invocation that selects only the frozen-"
        f"artifact gate; found {[(i, c) for i, _, c in artifact_routes]}"
    )
    _, gate_step, gate_command = artifact_routes[0]
    assert "shell" not in gate_step, (
        "frozen-artifact gate must inherit pwsh; shell override forbidden"
    )
    assert "if" not in gate_step, (
        "the frozen-artifact gate step must be unconditional and may not "
        "declare a disabling or dynamic if condition"
    )
    assert gate_step.get("continue-on-error", False) is False, (
        "the frozen-artifact gate must not use true or dynamic "
        "continue-on-error; every failure must block the job"
    )
    assert _ARTIFACT_PYTEST_INVOCATION.fullmatch(gate_command), (
        "artifact pytest must be one standalone PowerShell statement with "
        "the approved test, marker, timeout, and JUnit arguments; command "
        f"tails or control operators are forbidden: {gate_command!r}"
    )
    assert "test_real_frozen_bundle.py" in gate_command, (
        f"artifact selector is not scoped to the frozen-bundle tests: {gate_command!r}"
    )

    junit_path = _command_junit_path(gate_command)
    assert junit_path is not None, (
        "post-build artifact pytest does not write JUnit, so skipped gates "
        "cannot be audited"
    )
    lines = _workflow_code_lines(gate_step)
    pytest_indices = [
        index for index, line in enumerate(lines) if line == gate_command
    ]
    assert len(pytest_indices) == 1, (
        f"artifact pytest command is not uniquely located in its step: {lines!r}"
    )
    pytest_index = pytest_indices[0]
    assert lines[:pytest_index] == [
        "New-Item -ItemType Directory -Force build | Out-Null"
    ], (
        "artifact pytest must have exactly the approved directory-creation "
        "prefix; earlier control flow or exit commands can bypass the gate"
    )
    assert pytest_index + 1 < len(lines) and _PYTEST_EXIT_GUARD.fullmatch(
        lines[pytest_index + 1]
    ), (
        "artifact pytest must be immediately followed by "
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; a guard before "
        "pytest or after another command does not propagate pytest's exact exit"
    )

    junit_index = pytest_index + 2
    assert junit_index < len(lines), "artifact pytest has no ordered JUnit skip audit"
    junit_load = re.fullmatch(
        r"\[xml\]\$result\s*=\s*Get-Content\s+-LiteralPath\s+"
        r"(?:\"([^\"]+)\"|'([^']+)'|(\S+))",
        lines[junit_index],
        flags=re.IGNORECASE,
    )
    assert junit_load is not None, (
        "the first command after pytest's exit guard must load its JUnit file"
    )
    audited_path = next(
        group for group in junit_load.groups() if group is not None
    )
    assert _normalized_workflow_path(audited_path) == _normalized_workflow_path(
        junit_path
    ), (
        f"skip audit reads JUnit {audited_path!r}, but artifact pytest writes "
        f"{junit_path!r}"
    )

    expected_skip_count = re.compile(
        r"\$skipped\s*=\s*@\(\$result\.SelectNodes\("
        r"(?:\"//testcase/skipped\"|'//testcase/skipped')"
        r"\)\)\.Count",
        flags=re.IGNORECASE,
    )
    expected_skip_guard = re.compile(
        r"if\s*\(\s*\$skipped\s+-ne\s+0\s*\)\s*\{",
        flags=re.IGNORECASE,
    )
    assert len(lines) > junit_index + 4, "JUnit skip audit is incomplete"
    assert expected_skip_count.fullmatch(lines[junit_index + 1]), (
        "JUnit audit must count actual //testcase/skipped nodes"
    )
    assert expected_skip_guard.fullmatch(lines[junit_index + 2]), (
        "JUnit audit must reject skipped != 0; inverted or impossible skip "
        "conditions do not guard the artifact gate"
    )
    assert re.match(r"throw\b", lines[junit_index + 3], flags=re.IGNORECASE), (
        "the skipped != 0 branch must throw"
    )
    assert lines[junit_index + 4] == "}", "the skipped != 0 throw block is not closed"

    return "\n".join(lines[junit_index:]), junit_path


# ---------------------------------------------------------------------------
# ci-prepare-vendor-stubs.ps1
# ---------------------------------------------------------------------------


def test_stubs_script_exists() -> None:
    assert _STUBS_SCRIPT.is_file(), (
        f"missing CI helper at {_STUBS_SCRIPT}"
    )


def test_stubs_script_stubs_llama_server_exe() -> None:
    """installer/build.ps1 explicitly throws on a missing
    llama-server.exe (see line 34-36 of build.ps1). If the stub
    list drops it, CI builds fail with 'Missing llama-server.exe'
    and the whole workflow is useless.

    Seeded-defect check: remove 'llama-server.exe' from the
    $stubFiles array in ci-prepare-vendor-stubs.ps1 -> this test
    fails.
    """
    text = _STUBS_SCRIPT.read_text(encoding="utf-8")
    assert "llama-server.exe" in text, (
        "ci-prepare-vendor-stubs.ps1 does not list llama-server.exe. "
        "build.ps1 throws without it; CI will fail on every run."
    )


def test_stubs_script_is_idempotent_over_existing_files() -> None:
    """Local dev boxes keep real llama-server.exe + DLLs in
    third_party/llama.cpp/. The stub script must not clobber them --
    running ci-prepare-vendor-stubs.ps1 locally should be a no-op
    when the real binaries are already present.

    Seeded-defect check: change the `if (Test-Path $target) { continue }`
    guard to unconditional overwrite -> real binaries get replaced
    with empty stubs next time someone runs this script locally.
    """
    text = _STUBS_SCRIPT.read_text(encoding="utf-8")
    assert "Test-Path $target" in text, (
        "ci-prepare-vendor-stubs.ps1 lacks the skip-if-exists guard; "
        "will clobber real vendored binaries on dev machines"
    )


# ---------------------------------------------------------------------------
# ci-smoke.ps1
# ---------------------------------------------------------------------------


def test_smoke_script_exists() -> None:
    assert _SMOKE_SCRIPT.is_file(), f"missing CI helper at {_SMOKE_SCRIPT}"


def test_smoke_script_smokes_both_version_and_help() -> None:
    """ci-smoke.ps1 must exercise both --version AND --help from the
    INSTALLED exe. Dropping --help wouldn't make the CI red but
    would silently skip a class of bugs: help-text regressions where
    the string gets stripped or renamed.

    Asserts the INVOCATION pattern (the exe argument appears as a
    flag in an actual PS command), not just string presence in
    comments. Accepts either invocation style we've used:
      - `& $InstalledExe --version`          (direct & invocation)
      - `-CmdArgs @("--version")`            (Invoke-InstalledExe helper)
      - `-ArgumentList @("--version")`       (raw Start-Process)

    Seeded-defect check: replace the --help invocation block with a
    no-op / stub -> no `--help` appears in any invocation context,
    this test fails.
    """
    text = _SMOKE_SCRIPT.read_text(encoding="utf-8")
    # PowerShell invocation of the installed exe with each flag.
    # Match either the & style or the helper/Start-Process -CmdArgs style.
    version_pattern = (
        r"&\s*\$InstalledExe\s+--version"           # direct & invocation
        r"|-CmdArgs\s+@\([^)]*\"--version\"[^)]*\)"   # helper
        r"|-ArgumentList\s+@\([^)]*\"--version\""    # Start-Process
    )
    help_pattern = (
        r"&\s*\$InstalledExe\s+--help"
        r"|-CmdArgs\s+@\([^)]*\"--help\"[^)]*\)"
        r"|-ArgumentList\s+@\([^)]*\"--help\""
    )
    assert re.search(version_pattern, text), (
        "ci-smoke.ps1 does not invoke the installed exe with --version; "
        "install-layout bugs that the dev-venv tests miss won't surface"
    )
    assert re.search(help_pattern, text), (
        "ci-smoke.ps1 does not invoke the installed exe with --help; "
        "help-text regression would silently ship"
    )


def _smoke_invoke_helper_body() -> str:
    """The Invoke-InstalledExe function body from ci-smoke.ps1, with
    comment lines stripped so assertions see code, not prose. Slices
    from the function keyword to the first caller (the --version smoke
    Write-Host), which directly follows the function in the script."""
    text = _SMOKE_SCRIPT.read_text(encoding="utf-8")
    start = text.index("function Invoke-InstalledExe")
    end = text.index("==> Smoke: installed", start)
    body = text[start:end]
    return "\n".join(
        ln for ln in body.splitlines() if not ln.lstrip().startswith("#")
    )


def test_smoke_invoke_helper_enforces_timeout_via_waitforexit() -> None:
    """Invoke-InstalledExe declares -TimeoutSec, but the original body
    waited via ``Start-Process -Wait`` -- which blocks with NO timeout,
    making $TimeoutSec dead code (-TimeoutSec declared but unenforced).
    A frozen-import deadlock in the
    installed exe would hang the CI job until the runner's global
    6-hour timeout instead of failing in 30 s with a named cause.

    The body must launch without -Wait and enforce the budget itself:
    ``$proc.WaitForExit($TimeoutSec * 1000)``, killing the child on
    expiry.

    Seeded defect: re-add ``-Wait`` to the helper's Start-Process call,
    or delete the ``WaitForExit($TimeoutSec * 1000)`` enforcement ->
    this fails.
    """
    body = _smoke_invoke_helper_body()
    assert "Start-Process" in body, (
        "Invoke-InstalledExe no longer uses Start-Process; windowed-app "
        "exit-code capture contract broken (see ci-smoke.ps1 note)"
    )
    assert "-Wait" not in body, (
        "Invoke-InstalledExe passes -Wait to Start-Process: that blocks "
        "with no timeout, so -TimeoutSec is dead code and a hung exe "
        "hangs the whole CI job"
    )
    assert re.search(r"WaitForExit\(\s*\$TimeoutSec\s*\*\s*1000\s*\)", body), (
        "Invoke-InstalledExe never calls WaitForExit($TimeoutSec * 1000); "
        "the -TimeoutSec parameter is declared but unenforced"
    )
    assert ".Kill(" in body, (
        "Invoke-InstalledExe's timeout path does not Kill() the hung "
        "child; the orphan would outlive the smoke run and hold locks "
        "through the uninstall step"
    )


def test_smoke_invoke_helper_caches_handle_before_wait() -> None:
    """Without ``$null = $proc.Handle`` before WaitForExit, .NET may
    never cache the process handle and ``$proc.ExitCode`` is $null for
    a child that exits quickly -- turning the exit-code assertion into
    a comparison against $null that never fires (the same silent-pass
    Same handle-caching class as the WaitForExit timeout path.

    Seeded defect: delete the ``$null = $proc.Handle`` line -> this
    fails.
    """
    body = _smoke_invoke_helper_body()
    assert re.search(r"\$null\s*=\s*\$proc\.Handle", body), (
        "Invoke-InstalledExe does not touch $proc.Handle before "
        "WaitForExit; $proc.ExitCode can be $null for fast-exiting "
        "children and the exit-code check silently passes"
    )


def test_smoke_script_verifies_uninstall_cleanup() -> None:
    """The uninstall verification step is the whole reason the smoke
    script runs uninstall at all (vs just running install + launch).
    If the installer leaves files behind, users can't cleanly
    re-install or upgrade.

    Seeded-defect check: remove the Test-Path check on yamabiko.exe
    post-uninstall -> a broken uninstaller that leaves everything
    on disk would pass this CI without notice.
    """
    text = _SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert "Uninstall" in text, "ci-smoke.ps1 does not run uninstall"
    # Post-uninstall: must verify at least yamabiko.exe is gone.
    # The script's specific phrasing may evolve; grep for either the
    # Test-Path pattern against the expected-gone list, or the
    # $shouldBeGone array construction.
    assert (
        "shouldBeGone" in text
        and "yamabiko.exe" in text
        and "Test-Path" in text
    ), (
        "ci-smoke.ps1 lost its post-uninstall cleanup assertion; "
        "broken uninstallers will ship silently"
    )


def test_smoke_script_handles_nsis_noself_relocate() -> None:
    """NSIS uninstallers self-relocate to a temp dir at launch and
    re-exec, which breaks `Start-Process -Wait` since the original
    process exits before cleanup finishes. The `_?=` argument
    disables self-relocation, making -Wait meaningful.

    Seeded-defect check: remove the `_?=` from the uninstall
    invocation -> the script's `-Wait` returns before uninstall
    actually finishes, subsequent file-existence asserts race the
    unlink.
    """
    text = _SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert "_?=" in text, (
        "ci-smoke.ps1 invokes Uninstall.exe without NSIS _?= -- "
        "post-uninstall cleanup will race the self-relocated "
        "uninstaller process on CI, producing flaky failures"
    )


def test_smoke_script_proves_running_process_guards_block_and_scope() -> None:
    """Issue #21/#23(c): the smoke must (1) start a fake yamabiko.exe and
    assert BOTH installer and uninstaller abort with NSIS script-abort
    exit code 2, (2) start a fake llama-server.exe UNDER the install dir
    and assert the installer aborts with 2, and (3) start a fake
    llama-server.exe OUTSIDE the install dir and assert the installer
    still succeeds (path-scoping negative control -- an LM Studio /
    ollama llama-server must never block our install).

    Seeded-defect checks: delete any scenario block -> the matching
    assert below fails. NSI-side seeded defect (manual drill, RELEASE.md
    row 9): comment out the `Call EnsureNotRunning` inside
    `${If} ${Silent}` in .onInit -> Scenario B1's exit-code-2 assert
    throws on a live run. The smoke is silent-only, so the interactive
    call in WelcomeLeave is exercised by the release's installer-pages
    drive, not here.
    """
    text = _SMOKE_SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "Start-FakeProcess" in code, (
        "ci-smoke.ps1 has no fake-process helper; running-process guards "
        "are untested"
    )
    # Both blocked paths must assert the NSIS script-abort code exactly.
    assert len(re.findall(r"\.ExitCode\s+-ne\s+2", code)) >= 3, (
        "ci-smoke.ps1 does not assert exit code 2 on all three blocked "
        "paths (install-blocked-by-yamabiko, uninstall-blocked-by-"
        "yamabiko, install-blocked-by-scoped-llama)"
    )
    assert "yamabiko-ci-fake" in code and "llama-server.exe" in code, (
        "ci-smoke.ps1 lost a fake-process scenario (yamabiko.exe image "
        "probe or llama-server path-scoped probe)"
    )
    assert "yamabiko-ci-other-llama" in code, (
        "ci-smoke.ps1 lost the negative control: a llama-server.exe "
        "running OUTSIDE the install dir must NOT block the installer "
        "(path-scoping negative control)"
    )


def test_smoke_script_restores_start_menu_shortcuts() -> None:
    """The installer overwrites and the uninstaller deletes the shortcuts
    in the Start Menu's Yamabiko folder; on a dev box those belong to the
    real install, and RELEASE.md's seeded drill runs this script locally.
    The smoke must copy the folder aside before the first install and put
    it back from its outer finally block, which runs on success, on a
    throw and on -KeepInstall.

    Seeded defect: delete the Start Menu restore from the finally block, or
    move the copy below the first install -> a local run deletes (or
    "restores" the smoke's own) shortcuts -> this fails. Move the Start
    Menu restore out of its nested finally, next to the key restore -> a
    throwing key restore skips it -> this fails too.
    """
    text = _SMOKE_SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )
    assert '[Environment]::GetFolderPath("Programs")' in code, (
        "ci-smoke.ps1 does not locate the per-user Start Menu Programs folder"
    )
    backup = re.search(r"Copy-Item\s+-Recurse\s+\$StartMenuDir\s+\$StartMenuBackup", code)
    install = code.index('@("/S", "/D=$DestDir")')
    assert backup and backup.start() < install, (
        "the Start Menu folder is not copied aside before the first install"
    )
    outer_finally = re.search(r"(?m)^finally\s*\{", code)
    assert outer_finally, "ci-smoke.ps1 has no outer finally block"
    tail = code[outer_finally.start():]
    assert re.search(r"Remove-Item\s+-Recurse\s+-Force\s+\$StartMenuDir", tail), (
        "the finally block does not remove the smoke's Start Menu folder"
    )
    assert re.search(r"Copy-Item\s+-Recurse\s+\$StartMenuBackup\s+\$StartMenuDir", tail), (
        "the finally block does not put the copied Start Menu folder back"
    )
    key_restore = tail.find("-Action restore")
    nested = re.compile(r"finally\s*\{").search(tail, max(key_restore, 0))
    menu_restore = re.search(r"Remove-Item\s+-Recurse\s+-Force\s+\$StartMenuDir", tail)
    assert key_restore >= 0 and nested and nested.start() < menu_restore.start(), (
        "the Start Menu restore is not in a finally block after the key "
        "restore; a throwing key restore would skip it"
    )


def _closing_brace(code: str, at: int) -> int:
    """The index of the ``}`` that closes the ``{`` at ``code[at]``."""
    depth = 0
    for i in range(at, len(code)):
        depth += {"{": 1, "}": -1}.get(code[i], 0)
        if depth == 0:
            return i
    raise AssertionError(f"unbalanced braces after offset {at}")


def test_smoke_script_waits_for_its_installer_before_restoring() -> None:
    """Ctrl+C (a pipeline stop) ends a wait on the smoke's installer or
    uninstaller but not the child itself, and the outer finally block runs
    at once. Restoring then puts the real key and shortcuts back and
    deletes the snapshots, and the still running child overwrites or
    deletes them afterwards, with nothing left to restore from. Every
    installer and uninstaller run must go through the helper that records
    its process before waiting on it, and the finally block must wait for
    the recorded children and restore nothing while one still runs.

    Seeded defect: launch an installer with ``Start-Process ... -Wait``
    again, record the child only after waiting on it, restore before the
    wait, restore without checking that no child still runs, or restore
    the Start Menu (its copy, or the absence marker's cleanup) outside the
    branch for no child running (in the other branch, or below both) ->
    this fails.
    """
    text = _SMOKE_SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )
    launches = re.findall(
        r"(\S+)\s+-FilePath\s+\$(?:Installer\.FullName|InstalledUninst)\b", code
    )
    assert launches and set(launches) == {"Invoke-SmokeChild"}, (
        f"an installer or uninstaller run bypasses Invoke-SmokeChild: {launches}"
    )
    helper = re.search(r"(?ms)^function Invoke-SmokeChild\b.*?^\}", code)
    assert helper, "ci-smoke.ps1 has no Invoke-SmokeChild helper"
    record = helper.group(0).find("$SmokeChildren.Add(")
    assert 0 <= record < helper.group(0).find(".WaitForExit("), (
        "Invoke-SmokeChild does not record the child before waiting on it; "
        "a stop during the wait leaves the finally block blind to it"
    )
    outer_finally = re.search(r"(?m)^finally\s*\{", code)
    assert outer_finally, "ci-smoke.ps1 has no outer finally block"
    tail = code[outer_finally.start():]
    wait = re.search(r"\$StillRunning\s*=\s*@\(Wait-SmokeChildren\b", tail)
    guard = re.search(
        r"if\s*\(\s*\$StillRunning\.Count\s+-eq\s+0\s*\)\s*(\{)\s*try\s*\{", tail
    )
    restore = tail.find("-Action restore")
    assert wait and guard and wait.start() < guard.start() < restore, (
        "the finally block restores without first waiting for the smoke's "
        "installer or uninstaller and checking that none still runs"
    )
    guarded_end = _closing_brace(tail, guard.start(1))
    guarded = tail[guard.start(1):guarded_end]
    copy_back = re.search(r"Copy-Item\s+-Recurse\s+\$StartMenuBackup\s+\$StartMenuDir", guarded)
    drop_copy = re.search(r"Remove-Item\s+-Recurse\s+-Force\s+\$StartMenuBackup", guarded)
    drop_marker = re.search(r"Remove-Item\s+\$StartMenuAbsent\b", guarded)
    assert copy_back and drop_copy and drop_marker, (
        "the Start Menu restore is not in the branch for no child running; "
        "it would put the folder back and delete its copy (or delete the "
        "absence marker) while the smoke's installer or uninstaller still runs"
    )
    still_running = re.match(r"\s*else\s*\{", tail[guarded_end + 1:])
    assert still_running, "the wait has no branch for a child still running"
    else_open = guarded_end + still_running.end()
    else_body = tail[else_open + 1:_closing_brace(tail, else_open)]
    assert not re.search(r"Copy-Item|Remove-Item|-Action\s+restore", else_body), (
        "the branch for a child still running restores or deletes something"
    )


def test_smoke_recovery_steps_parse_with_apostrophes_in_paths(tmp_path: Path) -> None:
    """The refusal and the still-running message print, for each snapshot,
    the PowerShell command that puts its state back. Windows allows an
    apostrophe in a user name (C:\\Users\\O'Brien), so %TEMP% and the Start
    Menu folder can hold one, and PowerShell reads the typographic quotes
    (’) as single quotes too. Every printed command must still parse, with
    each path as one literal.

    Seeded defect: put a path inside '...' without doubling its quotes, or
    double only the ASCII apostrophe -> this fails.
    """
    odd = tmp_path / "O'Brien’s"
    odd.mkdir()
    paths = {
        "P_HELPER": str(odd / "snapshot-registry-key.ps1"),
        "P_KEY": r"HKCU:\Software\O'Brien’s\Yamabiko",
        "P_REG": str(odd / "uninstallkey.reg"),
        "P_MENU": str(odd / "Programs" / "Yamabiko"),
        "P_BACKUP": str(odd / "startmenu"),
        "P_ABSENT": str(odd / "startmenu.absent"),
    }
    # One snapshot of each kind, so every step is printed.
    (odd / "uninstallkey.reg").write_text("", encoding="ascii")
    (odd / "startmenu").mkdir()
    (odd / "startmenu.absent").write_text("", encoding="ascii")
    probe = tmp_path / "probe.ps1"
    probe.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
        "$P = [System.Management.Automation.Language.Parser]\n"
        "$smoke = $P::ParseFile($env:SMOKE_SCRIPT, [ref]$null, [ref]$null)\n"
        "foreach ($f in $smoke.FindAll({ $args[0] -is "
        "[System.Management.Automation.Language.FunctionDefinitionAst] }, $true)) {\n"
        "    . ([ScriptBlock]::Create($f.Extent.Text))\n"
        "}\n"
        "$SnapshotHelper = $env:P_HELPER; $AppUninstallKey = $env:P_KEY\n"
        "$RegSnapshot = $env:P_REG; $StartMenuDir = $env:P_MENU\n"
        "$StartMenuBackup = $env:P_BACKUP; $StartMenuAbsent = $env:P_ABSENT\n"
        "$steps = @(Get-RestoreSteps | ForEach-Object {\n"
        "    $errors = $null\n"
        "    $ast = $P::ParseInput($_, [ref]$null, [ref]$errors)\n"
        "    [pscustomobject]@{\n"
        "        step = $_\n"
        "        errors = @($errors | ForEach-Object Message)\n"
        "        literals = @($ast.FindAll({ $args[0] -is "
        "[System.Management.Automation.Language.StringConstantExpressionAst] }, $true)"
        " | ForEach-Object Value)\n"
        "    }\n"
        "})\n"
        "ConvertTo-Json -InputObject $steps -Depth 3\n",
        encoding="ascii",
    )
    result = subprocess.run(
        ["pwsh.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
         "-ExecutionPolicy", "Bypass", "-File", str(probe)],
        env={**os.environ, "SMOKE_SCRIPT": str(_SMOKE_SCRIPT), **paths},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    steps = json.loads(result.stdout)
    wanted = [
        {paths["P_HELPER"], paths["P_KEY"], paths["P_REG"]},
        {paths["P_MENU"], paths["P_BACKUP"]},
        {paths["P_MENU"], paths["P_ABSENT"]},
    ]
    assert len(steps) == len(wanted), [s["step"] for s in steps]
    for step, paths_in_step in zip(steps, wanted):
        assert not step["errors"], (
            f"a printed recovery step doesn't parse: {step['step']}\n{step['errors']}"
        )
        missing = paths_in_step - set(step["literals"])
        assert not missing, (
            f"a printed recovery step doesn't name {missing} as one literal: {step['step']}"
        )


def test_smoke_script_refuses_leftover_snapshots_before_deleting_the_test_install() -> None:
    """A run that finds an earlier run's snapshots refuses before anything
    changes: the earlier run's installer or uninstaller may still be
    running in the test install, and the snapshots hold the real state.
    The refusal must come before the pre-clean deletes ``$DestDir`` and
    before the uninstall key snapshot is saved over.

    Seeded defect: move the ``$DestDir`` pre-clean (or the snapshot save)
    back above the leftover refusal -> this fails.
    """
    text = _SMOKE_SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )
    refusal = code.find('throw ("An earlier smoke run left its snapshots')
    assert refusal >= 0, "ci-smoke.ps1 lost its leftover-snapshot refusal"
    pre_clean = re.search(r"Remove-Item\b[^\n]*\$DestDir\b", code)
    save = code.find("-Action save")
    assert pre_clean and save >= 0, (
        "ci-smoke.ps1 lost its $DestDir pre-clean or its snapshot save"
    )
    assert refusal < pre_clean.start(), (
        "the pre-clean deletes the test install before the leftover refusal; "
        "a refused run has already changed state"
    )
    assert refusal < save, (
        "the uninstall key snapshot is saved before the leftover refusal; "
        "it overwrites the earlier run's snapshot of the real key"
    )


def test_smoke_fake_process_uses_os_system_directory_not_optional_windir() -> None:
    """The smoke runs in shells where WINDIR may be absent.

    Seeded defect: resolve cmd.exe through ``$env:WINDIR`` again. An empty
    variable turns the source into ``\\System32\\cmd.exe`` on the current drive,
    so every running-process scenario fails before it tests the installer.
    """
    text = _SMOKE_SCRIPT.read_text(encoding="utf-8")
    start = text.index("function Start-FakeProcess")
    end = text.index("function Stop-FakeProcess", start)
    helper = text[start:end]

    assert "[Environment]::SystemDirectory" in helper
    assert "$env:WINDIR" not in helper


def test_smoke_script_reinstalls_over_existing_and_purges_stale_internal() -> None:
    """Issue #23(b): the smoke must run the installer a SECOND time over
    the live install with a stale sentinel planted in _internal\\, and
    assert the sentinel is purged (yamabiko.nsi's pre-clean postcondition,
    the DLL-shadowing class via the upgrade path) and the
    reinstalled exe still answers --version.

    Seeded-defect check: delete the reinstall scenario block from
    ci-smoke.ps1 (or drop its stale-sentinel Test-Path assert) -> this
    fails. NSI-side seeded defect (manual drill, RELEASE.md rotation):
    comment out `RMDir /r "$INSTDIR\\_internal"` in yamabiko.nsi ->
    the scenario itself throws on the surviving sentinel.
    """
    text = _SMOKE_SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "_internal\\stale-from-previous-version" in code, (
        "ci-smoke.ps1 never plants a stale-version sentinel inside "
        "_internal\\ before reinstalling (a sentinel at $DestDir root "
        "would not exercise the RMDir /r pre-clean); the upgrade "
        "pre-clean contract is untested"
    )
    # The sentinel must be asserted GONE after the second install run.
    assert re.search(
        r"Test-Path\s+\$StaleSentinel", code
    ), (
        "ci-smoke.ps1 plants the stale sentinel but never checks it was "
        "purged by the reinstall"
    )
    # And there must actually BE a second install invocation (same /S /D=).
    assert len(re.findall(r'@\("/S",\s*"/D=\$DestDir"\)', code)) >= 2, (
        "ci-smoke.ps1 only invokes the installer once; "
        "reinstall-over-existing is not exercised"
    )


# ---------------------------------------------------------------------------
# installer-smoke.yml
# ---------------------------------------------------------------------------


def test_workflow_exists() -> None:
    assert _WORKFLOW.is_file(), f"missing CI workflow at {_WORKFLOW}"


def test_workflow_is_valid_yaml() -> None:
    """YAML syntax check. A malformed workflow file is silently
    ignored by GitHub Actions (no run at all), so the first signal
    would be "CI stopped running" -- easy to miss on a quiet branch.

    Note: PyYAML under YAML 1.1 semantics parses `on:` as the
    boolean True (the `!!bool` schema accepts on/off/yes/no). We
    load, then confirm the top-level structure is a dict.
    """
    import yaml
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(data, dict), (
        f"workflow did not parse to a mapping; got {type(data).__name__}"
    )
    # `on` parsed as True by PyYAML; `jobs` stays a string key.
    triggers_key = True if True in data else "on"
    assert triggers_key in data, (
        "workflow missing top-level `on:` triggers block"
    )
    assert "jobs" in data, "workflow missing `jobs:` block"


def test_workflow_triggers_on_main_prs_pushes_and_v_tags() -> None:
    """The release gate must fire on PRs targeting main, pushes to
    main, and v* tags (the project's v-tag release path). The previous body
    only checked the event KEYS existed — editing ``branches: [main]``
    to a branch that never exists would silently detach the gate while
    the test stayed green (name-body drift: the branch filter changed
    but the assertion only checked key existence).
    Mirror the ci.yml sibling (test_ci_workflow_triggers_on_dev_and_main)
    and assert the branch/tag filters themselves.

    Seeded-defect check: point pull_request/push ``branches:`` away
    from main, or drop the ``v*`` entry from push ``tags:`` -> this
    test fails.
    """
    import yaml
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    triggers = data[True] if True in data else data.get("on", {})
    for event in ("pull_request", "push"):
        assert event in triggers, (
            f"workflow missing `{event}:` trigger; the release gate "
            f"would never fire on that path"
        )
        branches = (triggers[event] or {}).get("branches", [])
        assert "main" in branches, (
            f"installer-smoke {event} branch filter {branches!r} does "
            f"not include `main`; the release gate silently stops "
            f"firing on the release branch"
        )
    tags = (triggers["push"] or {}).get("tags", [])
    assert "v*" in tags, (
        f"installer-smoke push tag filter {tags!r} lacks 'v*'; v-tag "
        f"release builds would skip the gate"
    )


def test_workflow_declares_expected_jobs() -> None:
    """Post-split (commit e9da89c), installer-smoke.yml owns ONLY the
    heavyweight ``installer-smoke`` job (build + install + launch +
    uninstall). The fast ``unit-tests`` + ``real-tier-tests`` jobs
    moved to ci.yml -- asserted by
    test_ci_workflow_declares_fast_feedback_jobs. The installer-smoke
    job is the heavyweight release-gate.

    Seeded-defect check: delete the installer-smoke job -> the whole
    release-gate workstream is gone. This test fails.
    """
    import yaml
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    jobs = data["jobs"]
    expected = {"installer-smoke"}
    missing = expected - jobs.keys()
    assert not missing, f"installer-smoke.yml missing jobs: {missing}"


def test_workflow_installer_smoke_runs_bundle_tests() -> None:
    """The installer-smoke job must run tests/test_real_frozen_bundle.py
    against the freshly-built bundle. Skipping these leaves no guard on
    the missing-dependency crash classes caught only by bundle tests
    (e.g. missing CUDA DLLs, broken import graph).

    Seeded-defect check: remove the `pytest tests/test_real_frozen_bundle.py`
    step from the workflow -> installer smoke passes even when
    CUDA DLLs are missing or the import graph is broken.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "test_real_frozen_bundle.py" in text, (
        "installer-smoke job does not run test_real_frozen_bundle.py; "
        "frozen-bundle missing-dependency crash classes will ship unchecked"
    )


def test_workflow_installer_smoke_calls_ci_smoke_script() -> None:
    """The install + launch + uninstall cycle is delegated to
    installer/ci-smoke.ps1 so the same logic runs locally and in CI.
    If the workflow inlines its own install/launch/uninstall
    instead, the local-CI parity is lost.

    Seeded-defect check: replace the ``installer/ci-smoke.ps1``
    step with inline PowerShell -> script reference drops out of
    the workflow, this test fails.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "ci-smoke.ps1" in text, (
        "installer-smoke job doesn't invoke ci-smoke.ps1; CI and "
        "local smoke will drift"
    )
    assert "ci-prepare-vendor-stubs.ps1" in text, (
        "installer-smoke job doesn't stub the vendored llama "
        "binaries; build.ps1 will throw on missing llama-server.exe"
    )


def test_workflow_uploads_runtime_log_artifacts_on_failure() -> None:
    """On CI failure, a human needs the per-process runtime logs
    (main-watchdog.log / main-pipeline.log / main-overlay.log, one per
    process role) to diagnose without reproducing locally.

    The previous body asserted ``"main" in text and "log" in text``
    over the raw workflow file -- evergreen, because ``branches: [main]``
    and the word "logs" elsewhere satisfied it even with the upload
    step deleted (structural liar). Walk the parsed steps instead, the
    same structured technique
    test_installer_smoke_artifacts_marked_not_shippable already uses.

    Seeded-defect checks:
      * delete the failure-gated upload-artifact step whose path globs
        the collected ``*.log`` files -> this fails;
      * delete the failure-gated collect step that copies
        ``%LOCALAPPDATA%\\Yamabiko\\logs`` into runner.temp -> this
        fails (the upload would silently grab nothing).
    """
    import yaml
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = data["jobs"]["installer-smoke"]["steps"]
    failure_steps = [
        s for s in steps
        if isinstance(s, dict) and "failure()" in str(s.get("if", ""))
    ]
    assert failure_steps, (
        "workflow has no failure-conditional steps at all; CI failures "
        "become mysteries"
    )

    # 1. A failure-gated upload-artifact step must capture .log files.
    log_uploads = [
        s for s in failure_steps
        if "upload-artifact" in str(s.get("uses", ""))
        and re.search(r"\.log\b", str(s.get("with", {}).get("path", "")))
    ]
    assert log_uploads, (
        "no failure-gated upload-artifact step captures *.log files; "
        "the primary diagnostic source (per-process main-*.log) won't "
        "reach the artifact list"
    )

    # 2. The collect step feeding that upload must source the real app
    #    log dir (%LOCALAPPDATA%\Yamabiko\logs). Without it the upload
    #    globs an empty runner.temp dir and uploads nothing.
    collectors = [
        s for s in failure_steps
        if "Yamabiko" in str(s.get("run", "")) and "logs" in str(s.get("run", ""))
    ]
    assert collectors, (
        "no failure-gated run step copies %LOCALAPPDATA%\\Yamabiko\\logs "
        "to a path the upload-artifact step can reach; the log upload "
        "would silently capture nothing"
    )


def test_workflow_dep_audit_audits_release_lock_without_resolution() -> None:
    """The weekly dep-audit job must run pip-audit against
    constraints-release.txt (the fully-pinned lock the release install
    sequence actually resolves), with --no-deps (audit exactly the
    lock's pins, no re-resolution) and --disable-pip (skip pip's
    resolver entirely; also what lets the audit run without resolving
    the Windows-only PyAudioWPatch wheel). Auditing the requirements
    RANGES instead would resolve to the NEWEST satisfying versions and
    stay green while a CVE against a pinned version ships.

    Seeded-defect checks (each alone goes red): point the pip-audit
    ``-r`` back at yamabiko/requirements.txt, drop --no-deps, or drop
    --disable-pip.
    """
    import yaml
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    jobs = data["jobs"]
    assert "dep-audit" in jobs, (
        "installer-smoke.yml lost the dep-audit job; the weekly CVE "
        "audit is gone"
    )
    audit_lines = [
        ln.strip()
        for s in jobs["dep-audit"].get("steps", [])
        if isinstance(s, dict)
        for ln in str(s.get("run", "")).splitlines()
        if re.match(r"pip-audit\b", ln.strip())
    ]
    assert audit_lines, (
        "dep-audit job has no pip-audit invocation (or it moved -- "
        "update this test alongside)"
    )
    for ln in audit_lines:
        assert re.search(r"-r\s+constraints-release\.txt", ln), (
            f"pip-audit line {ln!r} does not audit constraints-release.txt; "
            f"auditing requirement ranges resolves to the newest versions "
            f"and goes green over a CVE against a shipped pin"
        )
        assert "yamabiko/requirements" not in ln, (
            f"pip-audit line {ln!r} audits yamabiko/requirements.txt "
            f"ranges, not the release lock"
        )
        assert "--no-deps" in ln, (
            f"pip-audit line {ln!r} lacks --no-deps; pip-audit would "
            f"re-resolve the lock instead of auditing exactly its pins"
        )
        assert "--disable-pip" in ln, (
            f"pip-audit line {ln!r} lacks --disable-pip; the audit "
            f"invokes pip's resolver (and needs the Windows-only "
            f"PyAudioWPatch wheel to resolve)"
        )


def test_workflow_install_steps_guard_each_native_command() -> None:
    """pwsh multi-line run blocks only propagate the LAST command's
    $LASTEXITCODE, so a failed intermediate pip/python command is
    masked by a succeeding later one -- the job goes green over a
    broken install and whatever breaks downstream blames the wrong
    step. Every run block with 2+ native pip/python commands, in BOTH
    workflows, must follow each such command with an
    ``if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`` guard. The
    block's final line is exempt: the runner's pwsh wrapper appends an
    exit-on-$LASTEXITCODE for the last command, so only intermediate
    failures can be masked.

    Walks every job's steps by content (any qualifying run block,
    whatever the step is named), the same tolerant-YAML technique as
    test_ci_workflow_unit_job_sets_hf_hub_offline_tripwire.

    Seeded-defect check: delete any single guard line that follows an
    intermediate pip/python command in either workflow -> this fails
    naming the unguarded command.
    """
    import yaml
    guard = re.compile(
        r"if\s*\(\$LASTEXITCODE\s+-ne\s+0\)\s*\{\s*exit\s+\$LASTEXITCODE\s*\}"
    )
    native = re.compile(r"(python|pip)\b")
    checked_blocks = 0
    for wf in (_CI_WORKFLOW, _WORKFLOW):
        data = yaml.safe_load(wf.read_text(encoding="utf-8"))
        for job_name, job in data["jobs"].items():
            for step in job.get("steps", []):
                if not isinstance(step, dict):
                    continue
                lines = [
                    ln.strip()
                    for ln in str(step.get("run", "")).splitlines()
                    if ln.strip() and not ln.strip().startswith("#")
                ]
                natives = [i for i, ln in enumerate(lines) if native.match(ln)]
                if len(natives) < 2:
                    continue
                checked_blocks += 1
                for i in natives:
                    if i == len(lines) - 1:
                        continue  # final line: runner wrapper propagates rc
                    assert guard.match(lines[i + 1]), (
                        f"{wf.name} job {job_name!r} step "
                        f"{str(step.get('name', ''))!r}: native command "
                        f"{lines[i]!r} is not followed by a $LASTEXITCODE "
                        f"guard; its failure would be masked by the next "
                        f"command's exit code"
                    )
    # The pin must be pinning something: both workflows currently carry
    # multi-command install blocks. If this drops to zero the blocks were
    # restructured and this test needs updating alongside.
    assert checked_blocks >= 3, (
        f"only {checked_blocks} multi-command pwsh install block(s) found "
        f"across ci.yml + installer-smoke.yml; the guard pin has lost its "
        f"subject (blocks split up or moved -- update this test)"
    )


def test_workflow_has_concurrency_control() -> None:
    """Windows runner minutes are expensive. Cancelling in-progress
    runs when a new push lands on the same branch is table stakes.

    Seeded-defect check: remove the `concurrency:` block -> each
    push starts a fresh run on top of the previous, wasting CI time.
    """
    import yaml
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert "concurrency" in data, (
        "workflow lacks concurrency control; superseded runs will "
        "burn Windows-runner minutes"
    )


# ---------------------------------------------------------------------------
# ci.yml -- the lightweight dev/main gate split out by commit e9da89c
#
# Before the split, unit-tests + real-tier-tests lived inside
# installer-smoke.yml and ran daily on a schedule (burning the free
# Windows-runner minute budget). The split moved them to this fast
# workflow that fires per-commit on dev + main; installer-smoke.yml
# dropped to a weekly/release-gate cadence. These guards keep the
# split honest: a regression that drops a fast-feedback job or a
# trigger branch would otherwise leave dev work running with no CI.
# ---------------------------------------------------------------------------


def test_ci_workflow_exists_and_is_valid_yaml() -> None:
    """ci.yml is the fast unit + real-tier gate that runs on every
    dev/main PR + push. A malformed file is silently ignored by GitHub
    Actions (no run at all), so the first signal would be "CI stopped
    running" -- easy to miss on a quiet branch.

    Seeded-defect check: introduce a YAML syntax error in ci.yml ->
    safe_load raises / returns a non-dict; this test fails.
    """
    import yaml
    assert _CI_WORKFLOW.is_file(), f"missing CI workflow at {_CI_WORKFLOW}"
    data = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(data, dict), (
        f"ci.yml did not parse to a mapping; got {type(data).__name__}"
    )
    assert "jobs" in data, "ci.yml missing `jobs:` block"


def test_ci_workflow_declares_fast_feedback_jobs() -> None:
    """ci.yml must declare the two fast-feedback jobs the workflow
    split moved off installer-smoke.yml: ``unit-tests`` and
    ``real-tier-tests``. These give per-commit signal on dev without
    paying installer-smoke's ~45 min build.

    Seeded-defect check: delete unit-tests (or real-tier-tests) from
    ci.yml -> dev pushes lose their unit / real coverage silently.
    This test fails.
    """
    import yaml
    data = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    jobs = data["jobs"]
    expected = {"unit-tests", "real-tier-tests"}
    missing = expected - jobs.keys()
    assert not missing, f"ci.yml missing fast-feedback jobs: {missing}"


def test_ci_workflow_triggers_on_dev_and_main() -> None:
    """The whole point of the split: ci.yml fires on every PR + push
    to BOTH dev and main, so day-to-day dev work gets fast feedback
    and main keeps its unit/real gate. installer-smoke.yml only covers
    main.

    Seeded-defect check: drop `dev` from ci.yml's pull_request/push
    branch filters -> dev work runs with no CI at all. This fails.
    """
    import yaml
    data = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) parses the `on:` key as the boolean True.
    triggers = data[True] if True in data else data.get("on", {})
    for event in ("pull_request", "push"):
        assert event in triggers, f"ci.yml missing `{event}:` trigger"
        branches = triggers[event].get("branches", [])
        assert "dev" in branches, (
            f"ci.yml {event} does not include `dev`; dev work would "
            f"run without CI signal"
        )
        assert "main" in branches, (
            f"ci.yml {event} does not include `main`"
        )


def test_ci_workflow_unit_job_sets_hf_hub_offline_tripwire() -> None:
    """The unit-tests job must run its pytest step with HF_HUB_OFFLINE=1.
    That env is a tripwire, not an optimisation: twice on 2026-07-11 a
    unit test walked into a real 5.14 GiB GGUF download on the
    cache-less runner (the stranded-install heal probes the HF cache on
    every settings Apply). Offline mode turns any future leak of that
    class into an immediate named failure instead of a silent
    multi-gigabyte download. A refactor that drops the env line revives
    the whole class while every CI run stays green -- hence this
    structural pin.

    Seeded-defect check: delete the `HF_HUB_OFFLINE: "1"` env entry
    from ci.yml's unit-suite step (or the whole env block) -> this
    fails.
    """
    import yaml
    data = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    job = data["jobs"]["unit-tests"]
    unit_run_steps = [
        s for s in job.get("steps", [])
        if isinstance(s, dict) and "not real" in str(s.get("run", ""))
    ]
    assert unit_run_steps, (
        "unit-tests job has no step running the `pytest -m \"not real\"` "
        "suite; the unit tier itself is gone (or renamed -- update this "
        "test alongside)"
    )
    job_env = job.get("env") or {}
    for step in unit_run_steps:
        env = {**job_env, **(step.get("env") or {})}
        assert str(env.get("HF_HUB_OFFLINE", "")) == "1", (
            f"unit-suite step {step.get('name', step.get('run'))!r} does "
            f"not set HF_HUB_OFFLINE=1 (env: {env!r}); the unit tier can "
            f"reach the HF network again -- the five-GB-download-in-unit-"
            f"job class is unguarded"
        )


def test_workflow_marker_selections_exclude_artifact_prebuild_and_require_it_postbuild(
    tmp_path: Path,
) -> None:
    """Workflow selectors must route the built-candidate gate causally.

    The lightweight real job runs before any bundle exists, so its actual
    pytest expression must select the ordinary real sentinel and deselect the
    frozen-artifact sentinel. Conversely, an installer-smoke pytest command
    after the actual ``installer/build.ps1`` invocation must select only the
    frozen-artifact sentinel, emit JUnit, propagate pytest failures, and reject
    every skipped gate.

    Seeded defects: change the lightweight expression back to ``real`` (or to
    broad ``not frozen_artifact``), or change/remove/broaden the post-build
    artifact expression; the pytest collection probe selects the wrong real,
    artifact, or unmarked sentinel and this test fails. Removing the JUnit skip
    audit also fails independently.
    """
    import yaml

    ci = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    real_steps = ci["jobs"]["real-tier-tests"]["steps"]
    lightweight_commands = [
        command
        for step in real_steps
        if isinstance(step, dict)
        for command in _pytest_commands(step)
    ]
    assert len(lightweight_commands) == 1, (
        "real-tier-tests must have exactly one pytest command so its "
        f"pre-build selection is auditable; got {lightweight_commands}"
    )
    lightweight_expression = _marker_expression(lightweight_commands[0])
    assert lightweight_expression is not None, (
        f"lightweight real command has no -m selector: {lightweight_commands[0]!r}"
    )
    assert _selected_marker_probe_tests(tmp_path / "lightweight", lightweight_expression) == {
        "test_lightweight_real"
    }, (
        f"pre-build marker expression {lightweight_expression!r} must select "
        "ordinary real tests and exclude the built-candidate gate"
    )

    installer = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    _assert_installer_artifact_gate_contract(installer, tmp_path)


def test_artifact_skip_audit_behaviorally_accepts_zero_and_rejects_one(
    tmp_path: Path,
) -> None:
    """Execute the workflow's real audit against synthetic JUnit results.

    Seeded defect: invert ``$skipped -ne 0`` or make the throw branch
    unreachable. The one-skip invocation exits zero and this test fails even
    if all expected words remain in the YAML.
    """
    import yaml

    installer = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    audit_script, junit_path = _assert_installer_artifact_gate_contract(
        installer,
        tmp_path / "contract-probes",
    )
    report = tmp_path / Path(junit_path.replace("\\", "/"))
    report.parent.mkdir(parents=True, exist_ok=True)

    def run_audit(skipped: int) -> subprocess.CompletedProcess[str]:
        skip_node = "<skipped message='seeded skip'/>" if skipped else ""
        report.write_text(
            "<?xml version='1.0' encoding='utf-8'?>"
            f"<testsuites tests='1' skipped='{skipped}'>"
            f"<testsuite tests='1' skipped='{skipped}'>"
            f"<testcase classname='gate' name='artifact'>{skip_node}</testcase>"
            "</testsuite></testsuites>",
            encoding="utf-8",
        )
        return subprocess.run(
            [
                "pwsh.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                audit_script,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=15,
        )

    zero = run_audit(0)
    assert zero.returncode == 0, (
        f"zero-skip JUnit must pass the live audit: {zero.stdout}\n{zero.stderr}"
    )
    one = run_audit(1)
    assert one.returncode != 0, (
        "one-skip JUnit must fail the live audit; the skip guard is "
        f"decorative: stdout={one.stdout!r}, stderr={one.stderr!r}"
    )


def _replace_workflow_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, (
        f"seeded workflow mutation lost its unique target: {old!r}"
    )
    return text.replace(old, new, 1)


def _run_marker_contract_against_workflow(
    monkeypatch,
    tmp_path: Path,
    workflow_text: str,
) -> None:
    mutated_workflow = tmp_path / "installer-smoke.yml"
    mutated_workflow.write_text(workflow_text, encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "_WORKFLOW", mutated_workflow)
    test_workflow_marker_selections_exclude_artifact_prebuild_and_require_it_postbuild(
        tmp_path / "marker-probes"
    )


def test_workflow_contract_rejects_display_name_only_fake_build(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A display label is not proof that ``installer/build.ps1`` ran.

    Seeded defect: retain ``name: Build installer`` but replace its command
    with a log line. The artifact selector is still textually later, yet no
    candidate was built and the old name-based boundary false-greens.
    """
    text = _replace_workflow_once(
        _WORKFLOW.read_text(encoding="utf-8"),
        r"run: .\installer\build.ps1",
        "run: Write-Host 'build step intentionally does nothing'",
    )

    with pytest.raises(AssertionError, match="build.ps1"):
        _run_marker_contract_against_workflow(monkeypatch, tmp_path, text)


@pytest.mark.parametrize(
    "tail",
    [
        "; exit 0",
        "&& exit 0",
        "|| exit 0",
        "| Out-Null",
        "$(exit 0)",
    ],
    ids=[
        "statement-separator",
        "success-chain",
        "failure-chain",
        "pipeline",
        "subexpression",
    ],
)
@pytest.mark.parametrize(
    ("gate_line", "expected_error"),
    [
        (r"run: .\installer\build.ps1", r"standalone|build\.ps1|control|tail"),
        (
            "pytest tests/test_real_frozen_bundle.py -m frozen_artifact "
            "-v --timeout=60 --junitxml=build/frozen-artifact-gate.xml",
            r"standalone|pytest|control|tail",
        ),
    ],
    ids=["build", "artifact-pytest"],
)
def test_workflow_contract_rejects_gate_same_line_tail(
    monkeypatch,
    tmp_path: Path,
    tail: str,
    gate_line: str,
    expected_error: str,
) -> None:
    """Build and artifact pytest must each be one standalone statement.

    Seeded defects: append a statement separator, logical chain, pipeline, or
    subexpression to either causal command. A nonzero exit can otherwise be
    overwritten while marker/JUnit text and physical-line ordering remain.
    """
    text = _replace_workflow_once(
        _WORKFLOW.read_text(encoding="utf-8"),
        gate_line,
        f"{gate_line} {tail}",
    )

    with pytest.raises(AssertionError, match=expected_error):
        _run_marker_contract_against_workflow(monkeypatch, tmp_path, text)


@pytest.mark.parametrize(
    "invocation",
    [
        "./installer/build.ps1",
        r"& .\installer\build.ps1",
        "& './installer/build.ps1'",
        r'& ".\installer\build.ps1"',
    ],
    ids=["forward-slash", "call-operator", "quoted-posix", "quoted-windows"],
)
def test_workflow_contract_accepts_standalone_build_invocation_forms(
    monkeypatch,
    tmp_path: Path,
    invocation: str,
) -> None:
    """Slash, quote, and call-operator variants remain valid when standalone."""
    text = _replace_workflow_once(
        _WORKFLOW.read_text(encoding="utf-8"),
        r"run: .\installer\build.ps1",
        f"run: |\n          {invocation}",
    )

    _run_marker_contract_against_workflow(monkeypatch, tmp_path, text)


_INSTALLER_JOB_HEADER = "  installer-smoke:\n"
_BUILD_STEP_HEADER = "      - name: Build installer\n"
_ARTIFACT_STEP_HEADER = (
    "      - name: Run mandatory frozen-artifact gate with zero skips\n"
)
_BUILD_RUN_LINE = r"        run: .\installer\build.ps1" + "\n"
_ARTIFACT_RUN_PREFIX = (
    "        run: |\n"
    "          New-Item -ItemType Directory -Force build | Out-Null\n"
)
_PWSH_DEFAULTS = "defaults:\n  run:\n    shell: pwsh\n"


@pytest.mark.parametrize(
    ("old", "new", "expected_error"),
    [
        (
            _INSTALLER_JOB_HEADER,
            _INSTALLER_JOB_HEADER + "    if: ${{ false }}\n",
            "job|unconditional|if",
        ),
        (
            _BUILD_STEP_HEADER,
            _BUILD_STEP_HEADER + "        if: ${{ false }}\n",
            "unconditional|if",
        ),
        (
            _BUILD_STEP_HEADER,
            _BUILD_STEP_HEADER + "        continue-on-error: true\n",
            "continue-on-error|build",
        ),
        (
            _ARTIFACT_STEP_HEADER,
            _ARTIFACT_STEP_HEADER + "        if: ${{ false }}\n",
            "unconditional|if",
        ),
        (
            _ARTIFACT_STEP_HEADER,
            _ARTIFACT_STEP_HEADER + "        continue-on-error: true\n",
            "continue-on-error",
        ),
        (
            _ARTIFACT_RUN_PREFIX,
            _ARTIFACT_RUN_PREFIX.replace(
                "          New-Item",
                "          exit 0\n          New-Item",
            ),
            "prefix|control|pytest",
        ),
        (
            _BUILD_RUN_LINE,
            "        run: |\n          exit 0\n"
            "          .\\installer\\build.ps1\n",
            "build.ps1|invocation|command",
        ),
        (
            _BUILD_RUN_LINE,
            "        run: |\n          './installer/build.ps1'\n",
            "build.ps1|invocation",
        ),
        (
            _PWSH_DEFAULTS,
            "defaults:\n  run:\n    shell: cmd /c echo {0}\n",
            "shell|pwsh",
        ),
        (
            _INSTALLER_JOB_HEADER,
            _INSTALLER_JOB_HEADER + "    defaults:\n      run:\n"
            "        shell: cmd /c echo {0}\n",
            "shell|pwsh",
        ),
        (
            _BUILD_STEP_HEADER,
            _BUILD_STEP_HEADER + "        shell: cmd /c echo {0}\n",
            "shell|pwsh",
        ),
        (
            _ARTIFACT_STEP_HEADER,
            _ARTIFACT_STEP_HEADER + "        shell: cmd /c echo {0}\n",
            "shell|pwsh",
        ),
    ],
    ids=[
        "disabled-installer-job",
        "disabled-build",
        "continue-on-error-build",
        "disabled-artifact-gate",
        "continue-on-error-artifact-gate",
        "exit-zero-before-artifact-pytest",
        "exit-zero-before-build",
        "inert-quoted-build-string",
        "top-level-shell-override",
        "job-shell-override",
        "build-shell-override",
        "artifact-shell-override",
    ],
)
def test_workflow_contract_rejects_nonexecuting_or_nonblocking_gate_metadata(
    monkeypatch,
    tmp_path: Path,
    old: str,
    new: str,
    expected_error: str,
) -> None:
    """Build and artifact gates must execute unconditionally and block CI.

    Seeded defects: disable either step with ``if: ${{ false }}``, allow a
    build/artifact failure via ``continue-on-error: true``, or replace the
    build command with an inert quoted PowerShell string, or wrap a causal
    command in an inert custom shell template. Text and ordering remain, but
    no mandatory built-candidate gate exists.
    """
    text = _replace_workflow_once(
        _WORKFLOW.read_text(encoding="utf-8"),
        old,
        new,
    )

    with pytest.raises(AssertionError, match=expected_error):
        _run_marker_contract_against_workflow(monkeypatch, tmp_path, text)


def test_workflow_contract_rejects_failure_guard_before_pytest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The native exit-code guard must immediately follow artifact pytest.

    Seeded defect: move the otherwise-correct guard before pytest. Its text
    remains in the step but it observes an earlier command, so a failing
    artifact gate can be overwritten by the succeeding JUnit audit.
    """
    pytest_line = (
        "          pytest tests/test_real_frozen_bundle.py -m frozen_artifact "
        "-v --timeout=60 --junitxml=build/frozen-artifact-gate.xml\n"
    )
    guard_line = (
        "          if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
    )
    text = _replace_workflow_once(
        _WORKFLOW.read_text(encoding="utf-8"),
        pytest_line + guard_line,
        guard_line + pytest_line,
    )

    with pytest.raises(AssertionError, match="LASTEXITCODE|immediately"):
        _run_marker_contract_against_workflow(monkeypatch, tmp_path, text)


def test_workflow_contract_rejects_inverted_skip_comparison(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A skip condition that can never be true must not satisfy the audit.

    Seeded defect: invert ``$skipped -ne 0`` to ``$skipped -lt 0``. The old
    word-presence assertion still sees both ``skipped`` and ``throw`` while a
    one-skip JUnit result proceeds successfully.
    """
    text = _replace_workflow_once(
        _WORKFLOW.read_text(encoding="utf-8"),
        "if ($skipped -ne 0)",
        "if ($skipped -lt 0)",
    )

    with pytest.raises(AssertionError, match="skipped|zero"):
        _run_marker_contract_against_workflow(monkeypatch, tmp_path, text)


def test_workflow_contract_rejects_skip_audit_of_different_junit_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The skip audit must read the JUnit file emitted by artifact pytest.

    Seeded defect: point ``Get-Content`` at a stale report. Both JUnit and
    skip-audit words remain, but skipped artifact tests are no longer checked.
    """
    text = _replace_workflow_once(
        _WORKFLOW.read_text(encoding="utf-8"),
        "Get-Content -LiteralPath build/frozen-artifact-gate.xml",
        "Get-Content -LiteralPath build/stale-artifact-gate.xml",
    )

    with pytest.raises(AssertionError, match="JUnit|junit"):
        _run_marker_contract_against_workflow(monkeypatch, tmp_path, text)


# ---------------------------------------------------------------------------
# Shared llama vendor manifest (installer/llama-manifest.txt)
#
# The vendored llama-server.exe runs as a child process with its own DLL
# search path, so audit_bundle_dlls.py deliberately skips _internal/llama.
# Without a manifest, a re-vendor that drops a CUDA/ggml sidecar ships
# silently and the translator dies at startup. These guards keep the
# shared manifest the single source of truth for the stub list, the
# build-time assert, and the bundle-side preflight check.
# ---------------------------------------------------------------------------

_LLAMA_MANIFEST = _REPO_ROOT / "installer" / "llama-manifest.txt"
_BUILD_SCRIPT = _REPO_ROOT / "installer" / "build.ps1"
_REBUILD_SCRIPT = _REPO_ROOT / "installer" / "rebuild-fast.ps1"


def _manifest_entries() -> list[str]:
    return [
        ln.strip()
        for ln in _LLAMA_MANIFEST.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def test_llama_manifest_exists_and_lists_server() -> None:
    """The shared manifest must exist and at minimum list llama-server.exe
    plus its DLLs.

    Seeded defect: empty the manifest -> this fails.
    """
    assert _LLAMA_MANIFEST.is_file(), f"missing {_LLAMA_MANIFEST}"
    entries = _manifest_entries()
    assert "llama-server.exe" in entries, "manifest does not list llama-server.exe"
    assert len(entries) >= 2, "manifest lists no DLLs alongside the exe"


def test_build_scripts_assert_vendored_llama_against_manifest() -> None:
    """build.ps1 + rebuild-fast.ps1 must validate the vendored llama.cpp
    against the shared manifest and throw on a missing file -- not just
    check the .exe then blind-copy *.dll. The assert lives in the shared
    installer/assert-llama-vendor.ps1 (presence + PE-content), so the
    invariant splits: both build scripts invoke the helper, and the
    helper reads the manifest + throws.

    Seeded defect: drop the assert-llama-vendor.ps1 call from either
    script (reverting to a blind `Copy-Item *.dll`) -> this fails; gut
    the helper's manifest read or its throw-on-missing -> this fails.
    """
    for script in (_BUILD_SCRIPT, _REBUILD_SCRIPT):
        text = script.read_text(encoding="utf-8")
        assert "assert-llama-vendor.ps1" in text, (
            f"{script.name} does not invoke assert-llama-vendor.ps1; a "
            f"missing or stub vendored DLL would ship silently"
        )
    helper = _REPO_ROOT / "installer" / "assert-llama-vendor.ps1"
    helper_text = helper.read_text(encoding="utf-8")
    assert "llama-manifest.txt" in helper_text, (
        "assert-llama-vendor.ps1 does not read installer/llama-manifest.txt"
    )
    assert "missing" in helper_text.lower() and "throw" in helper_text, (
        "assert-llama-vendor.ps1 never throws on a missing vendored file"
    )


def test_build_scripts_use_start_process_for_version_smoke() -> None:
    """build.ps1 + rebuild-fast.ps1 must run the dist-exe --version
    smoke via Start-Process with explicit stdout redirection, an
    ExitCode check, AND a 'Yamabiko' stdout-marker assert -- not the
    bare ``&`` call operator. yamabiko.exe is a windowed
    (console=False) PyInstaller bootloader: under ``&`` the bootloader
    detaches from the parent console, stdout is discarded (the version
    string is never verified) and $LASTEXITCODE can be polluted by WER
    interaction on a bootloader crash -- the exact reason
    ci-smoke.ps1's Invoke-InstalledExe exists.

    Seeded defect: revert either script's smoke block to
    ``& (Join-Path $Dist "yamabiko.exe") --version`` +
    ``if ($LASTEXITCODE -ne 0) { throw ... }`` -> this fails.
    """
    for script in (_BUILD_SCRIPT, _REBUILD_SCRIPT):
        text = script.read_text(encoding="utf-8")
        code = "\n".join(
            ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
        )
        assert not re.search(
            r"&\s*\(Join-Path[^\n]*yamabiko\.exe[^\n]*--version", code
        ), (
            f"{script.name} still smokes --version via the bare `&` call "
            f"operator; for a windowed bootloader that discards stdout "
            f"and can miss a crashed child"
        )
        assert "Start-Process" in code, (
            f"{script.name} has no Start-Process invocation for the "
            f"--version smoke"
        )
        assert re.search(r'-ArgumentList\s+@\("--version"\)', code), (
            f"{script.name} does not pass --version via -ArgumentList; "
            f"the smoke no longer exercises the version path"
        )
        assert "-RedirectStandardOutput" in code, (
            f"{script.name} does not redirect the smoke's stdout; the "
            f"version string can't be verified for a windowed exe"
        )
        assert re.search(r"\.ExitCode\s+-ne\s+0", code), (
            f"{script.name} never checks the smoke child's .ExitCode; a "
            f"crashing bootloader would pass the build"
        )
        assert re.search(r'-notmatch\s+"Yamabiko"', code), (
            f"{script.name} never asserts the 'Yamabiko' marker in the "
            f"--version stdout; an exe that exits 0 with empty output "
            f"would pass the build"
        )
        assert re.search(r"WaitForExit\(", code), (
            f"{script.name} waits on the smoke child without a timeout; "
            f"a frozen-import deadlock would hang the build forever "
            f"(frozen-import deadlock class)"
        )


def test_build_script_verifies_venv_against_release_lock_before_pyinstaller() -> None:
    """build.ps1 must COMPARE the build venv's installed versions against
    constraints-release.txt (via `pip freeze`, the command that generates
    the lock) and throw on any divergence, BEFORE PyInstaller runs. The
    lock used to live only in build.ps1's Prereqs comment: nothing checked
    the venv at build time, so one ad-hoc `pip install` after setup
    silently diverged the locally-built installer -- the artifact users
    actually receive -- from the lock the CHANGELOG advertises.

    Comment lines are stripped first, so the Prereqs comment (which also
    names the lock) cannot satisfy the asserts.

    Seeded defects (each goes red): delete the verification block from
    build.ps1 (reverting the lock to a Prereqs-comment promise) -> the
    lock-name / pip-freeze / throw asserts fail; move the block below the
    PyInstaller invocation -> the ordering assert fails.
    """
    text = _BUILD_SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "constraints-release.txt" in code, (
        "build.ps1 no longer references constraints-release.txt outside "
        "comments; the release lock is a comment-only promise again"
    )
    assert "pip freeze" in code, (
        "build.ps1 never snapshots the venv via pip freeze; installed "
        "versions cannot be compared against the lock"
    )
    assert re.search(r"throw\s+\"[^\"]*diverges", code), (
        "build.ps1 never throws on venv/lock divergence; drift would "
        "print (at best) and the build would continue"
    )
    freeze_at = code.index("pip freeze")
    pyinstaller = re.search(
        r"(?im)^\s*&\s+\$Python\s+-m\s+PyInstaller\s+\$Spec\b",
        code,
    )
    assert pyinstaller is not None, (
        "build.ps1 no longer invokes PyInstaller through the selected "
        "release-venv Python and the resolved spec path"
    )
    pyinstaller_at = pyinstaller.start()
    assert freeze_at < pyinstaller_at, (
        "build.ps1 verifies the venv only AFTER PyInstaller has already "
        "frozen the divergent package set into dist/"
    )


def test_stub_script_uses_shared_manifest() -> None:
    """ci-prepare-vendor-stubs.ps1 must derive its stub list from the
    shared manifest, not an inline copy that can drift from what the build
    asserts/copies.

    Seeded defect: re-inline the cublas/ggml list -> this fails.
    """
    text = _STUBS_SCRIPT.read_text(encoding="utf-8")
    assert "llama-manifest.txt" in text, (
        "ci-prepare-vendor-stubs.ps1 does not read the shared manifest; "
        "the stub list can drift from build.ps1's copy list"
    )


def test_installer_smoke_artifacts_marked_not_shippable() -> None:
    """Every installer the smoke workflow produces is built with STUBBED
    llama binaries, so it cannot translate and must never be shipped. Any
    non-failure upload of yamabiko-*-setup.exe must be named to make that
    unmistakable (audit build-2: a stub-llama CI artifact could be grabbed
    and attached to a GitHub release -- shipping a brick).

    Seeded defect: rename an upload back to 'installer-${{ github.ref_name }}'
    -> this fails.
    """
    import yaml
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = data["jobs"]["installer-smoke"]["steps"]
    setup_uploads = [
        s for s in steps
        if isinstance(s, dict)
        and "upload-artifact" in str(s.get("uses", ""))
        and "setup.exe" in str(s.get("with", {}).get("path", ""))
    ]
    assert setup_uploads, "no setup.exe artifact uploads found (steps moved?)"
    for s in setup_uploads:
        if "failure()" in str(s.get("if", "")):
            continue  # failure-only diagnostic upload, clearly debug
        name = str(s.get("with", {}).get("name", "")).upper()
        assert "SMOKE" in name or "DO-NOT-SHIP" in name, (
            f"installer artifact {name!r} is not marked smoke-only/not-"
            f"shippable; a stub-llama installer could be mistaken for a "
            f"release"
        )


def test_smoke_script_asserts_user_data_survives_uninstall() -> None:
    """Issue #23(a): before uninstalling, the smoke must plant sentinel
    user data (user_overrides.yaml + an hf-cache file + a log) in the
    install dir, and after uninstall assert those files SURVIVE while
    the bundle files are gone. Without this, someone 'simplifying'
    yamabiko.nsi's final `RMDir "$INSTDIR"` to `RMDir /r` would nuke
    every real user's multi-GB model cache + settings and ship green.

    Seeded-defect check: drop the $mustSurvive assertion (or the
    sentinel planting) from ci-smoke.ps1 -> this fails; delete only
    the `if ($lost.Count...) { throw }` block while keeping the
    $mustSurvive/$lost lines -> the throw-phrase assert below goes
    red. NSI-side seeded defect (RELEASE.md rotation row 8): change
    the final `RMDir "$INSTDIR"` in yamabiko.nsi to
    `RMDir /r "$INSTDIR"` -> Scenario C throws on a live run.
    """
    text = _SMOKE_SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "mustSurvive" in code, (
        "ci-smoke.ps1 has no post-uninstall data-preservation assertion; "
        "an uninstaller that deletes user data would pass CI"
    )
    assert "user_overrides.yaml" in code and "hf-cache" in code, (
        "ci-smoke.ps1's data-preservation sentinels no longer cover the "
        "two highest-value user assets (config overrides + model cache)"
    )
    assert "DELETED user data it must preserve" in code, (
        "ci-smoke.ps1 lost the data-preservation throw (or its exact "
        "phrase) -- docs/RELEASE.md rotation row 8 quotes that string, "
        "and without the throw the $mustSurvive check is decorative"
    )


def test_requirements_dev_pins_release_tooling_deps() -> None:
    """pefile (scripts/audit_bundle_dlls.py / preflight section 9),
    psutil (tests/test_real_frozen_e2e.py), pytest itself, and
    pytest-timeout (preflight section 8's --timeout=180 + setup.cfg's
    mutmut runner --timeout=30) are hard deps of the release gate. They
    were split out of the runtime yamabiko/requirements.txt into the
    repo-root requirements-dev.txt (they have no production import); a
    fresh release box running only `pip install -r requirements-dev.txt`
    must still get all four, or the gate fails with a misleading error.

    Seeded defect: drop any of the four pins from requirements-dev.txt ->
    this fails.
    """
    req = (_REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert re.search(r"(?m)^\s*pefile\b", req), "requirements-dev.txt does not pin pefile"
    assert re.search(r"(?m)^\s*psutil\b", req), "requirements-dev.txt does not pin psutil"
    # NOTE: a bare \b after "pytest" would let `pytest-asyncio` /
    # `pytest-qt` satisfy the check (the word boundary sits before the
    # hyphen), so require a version operator / space / EOL instead.
    assert re.search(r"(?m)^\s*pytest(?:[<>=~! ]|$)", req), (
        "requirements-dev.txt does not pin pytest itself (only plugins); "
        "a fresh box gets pytest only as a transitive dep, unpinned"
    )
    assert re.search(r"(?m)^\s*pytest-timeout\b", req), (
        "requirements-dev.txt does not pin pytest-timeout; preflight "
        "section 8 passes --timeout=180 and exits 4 without it"
    )


def test_runtime_requirements_excludes_dev_tooling() -> None:
    """The runtime requirements file (read by pyproject's dynamic
    `dependencies`, so `pip install yamabiko` resolves it) must NOT list
    the six dev/release-gate tools -- none has a production import, and
    shipping them as runtime deps drags pytest into every install. They
    live in requirements-dev.txt instead.

    Seeded defect: leave `pytest>=8` (or any of the six) in
    yamabiko/requirements.txt -> this fails.
    """
    req = (_REPO_ROOT / "yamabiko" / "requirements.txt").read_text(encoding="utf-8")
    dev_tools = [
        "pytest", "pytest-asyncio", "pytest-qt", "pytest-timeout",
        "pefile", "psutil",
    ]
    # A requirement line starts at column 0 with the dist name followed by
    # a version operator, whitespace, or EOL. Comments (which mention the
    # names in the relocation note) start with '#', so `(?m)^\s*` anchored
    # to the name + operator won't match them.
    present = [
        name for name in dev_tools
        if re.search(rf"(?m)^{re.escape(name)}(?:[<>=~! ]|$)", req)
    ]
    assert not present, (
        f"yamabiko/requirements.txt still lists dev-only tooling as runtime "
        f"deps: {present}. Move them to requirements-dev.txt -- pyproject's "
        f"dynamic `dependencies` reads this file, so these would ship with "
        f"`pip install yamabiko`."
    )


def test_constraints_release_covers_every_release_install_package() -> None:
    """The release lock (constraints-release.txt) must pin every package
    the release install sequence resolves: the runtime
    yamabiko/requirements.txt names, the requirements-dev.txt tooling
    (the sequence's first command installs it under the lock), and
    pyinstaller itself (installed via `pip install "pyinstaller>=6.10"
    -c constraints-release.txt` with no requirements file naming it). A
    package in the sequence but absent from the lock resolves
    unconstrained -- a lock regenerated in a venv without pyinstaller
    would let the build tool float, unpinned bootloaders and all --
    silently reintroducing the non-reproducibility the lock exists to
    prevent.

    Compares PEP 503 canonical names so `PyYAML`/`pyyaml`,
    `hf_xet`/`hf-xet`, `SudachiDict-core`/`sudachidict_core` match.

    Seeded defects (each one line): add a new `some-dep>=1` line to
    either requirements file without regenerating the lock -> this
    fails; delete the `pyinstaller==` line from constraints-release.txt
    (a lock regenerated without pyinstaller installed) -> this fails.
    """
    from packaging.utils import canonicalize_name

    def _names(path: Path) -> set[str]:
        out: set[str] = set()
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", "-")):
                continue
            # Split on the first version operator / marker / comment.
            name = re.split(r"[<>=~!;# \t\[]", line, maxsplit=1)[0].strip()
            if name:
                out.add(canonicalize_name(name))
        return out

    required = _names(_REPO_ROOT / "yamabiko" / "requirements.txt")
    required |= _names(_REPO_ROOT / "requirements-dev.txt")
    required.add("pyinstaller")
    locked = _names(_REPO_ROOT / "constraints-release.txt")
    missing = required - locked
    assert not missing, (
        f"constraints-release.txt is missing these release-install deps: "
        f"{sorted(missing)}. Regenerate it from the build venv (with the "
        f"FULL release install sequence applied, pyinstaller included): "
        f".venv-release\\Scripts\\python.exe -m pip freeze --exclude-editable "
        f"> constraints-release.txt (then restore the header; see "
        f"docs/RELEASE.md, 'Release venv')."
    )


def test_pytest_timeout_plugin_is_importable_in_this_env() -> None:
    """Fail-fast companion: preflight section 8 and the
    setup.cfg mutmut runner both pass ``--timeout``; without the plugin
    every such invocation exits 4 ("unrecognized arguments") and the
    gate reports a misleading "real-path suite FAILED". This unit-tier
    test makes the gap fail FAST and legibly on any env that claims to
    run the suite.

    Seeded defect: `pip uninstall pytest-timeout` -> this fails with a
    message naming the actual problem.
    """
    import importlib.util
    assert importlib.util.find_spec("pytest_timeout") is not None, (
        "pytest-timeout is not installed in this environment; preflight "
        "section 8 (--timeout=180) and the mutmut runner (--timeout=30) "
        "will exit 4 with 'unrecognized arguments'. Fix: pip install -r "
        "requirements-dev.txt"
    )
