"""Unit tests for the static preflight sections added after the
install-surface audit, plus the scripts/ release-tooling helpers owned
by the same cluster (mutmut-analyze).

These sections are structural checks (grep + YAML + git) rather than
real-path probes, so they're fast to exercise in isolation — each test
constructs a temp directory with a fake ``yamabiko/``/``installer/``
layout, monkeypatches ``scripts.preflight.ROOT`` to point at it, and
invokes the section function directly.

Coverage:
- ``check_pyinstaller_spec`` — required faster-whisper package
  hiddenimport closure
- ``check_version_consistency`` — binary VERSIONINFO quad guard: repo
  files consistent / seeded quad + tuple + string drift / missing
  APP_VERSION_QUAD define
- ``check_config_key_consumers`` — pass / orphan / nested keys /
  collision-prone leaf names
- ``check_bundle_module_launchers`` — __main__ with dispatch, without
  dispatch, llama mismatch, empty-spec drift guard
- ``check_changelog_migration_notes`` — tag present / absent, config
  unchanged / changed-with-migration / changed-without-migration
- ``scripts/mutmut-analyze.py`` — dynamic section derivation + mutant
  accounting self-audit. Lives here because this is the cluster's owned
  test module; relocate to tests/test_mutmut_analyze.py if the scripts/
  tooling grows its own test home.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
from pathlib import Path

import pytest

from scripts import preflight


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """Minimal repo skeleton. Tests write just the files they need."""
    (tmp_path / "yamabiko").mkdir()
    (tmp_path / "installer").mkdir()
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Release build scripts: one pinned interpreter for every Python process
# ---------------------------------------------------------------------------

def _active_powershell_commands(source: str) -> list[str]:
    """Join backtick continuations while excluding blank/comment-only lines."""
    commands: list[str] = []
    command_parts: list[str] = []
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        continued = line.endswith("`")
        command_parts.append(line[:-1].rstrip() if continued else line)
        if not continued:
            commands.append(" ".join(command_parts))
            command_parts.clear()
    assert not command_parts, "PowerShell source ends in a dangling continuation"
    return commands


def test_build_scripts_use_one_explicit_venv_python_for_pyinstaller_and_helpers():
    """Both bundle paths stay inside one build venv from start to finish.

    A caller's PATH can point ``python`` and ``pyinstaller`` at different
    environments.  The scripts therefore pin one interpreter, the selected
    ``-Venv`` (default ``.venv``, resolved against the repo root), use its
    ``-m PyInstaller``, and invoke every Python release helper through it.

    Seeded defects: comment a required variable/argument, hard-code
    ``.venv`` again instead of ``$VenvDir``, change the ``-Venv`` default,
    select bare ``python`` for pip freeze, or remove a command's immediate
    exit guard. Active-command parsing makes each mutation observable instead
    of allowing comments or a later command's guard to satisfy the contract.
    """
    repo = Path(__file__).resolve().parents[1]
    script_paths = (
        repo / "installer" / "build.ps1",
        repo / "installer" / "rebuild-fast.ps1",
    )
    exact_variables = (
        '$VenvDir = if ([System.IO.Path]::IsPathRooted($Venv)) { $Venv } else { Join-Path $Repo $Venv }',
        '$Python = Join-Path $VenvDir "Scripts\\python.exe"',
        '$SitePackages = Join-Path $VenvDir "Lib\\site-packages"',
        '$Inventory = Join-Path $Repo "build\\yamabiko\\bundle-inventory.json"',
        '$ProvenanceConfig = Join-Path $Repo "installer\\bundle-provenance.toml"',
        '$ProvenanceReport = Join-Path $Repo "build\\yamabiko\\bundle-provenance.json"',
        '$LicenseOverrides = Join-Path $Repo "installer\\licenses-manifest.txt"',
        '$LicenseIndex = Join-Path $Dist "_internal\\licenses\\INDEX.json"',
        '$ClosureAudit = Join-Path $Repo "build\\yamabiko\\license-closure-audit.json"',
        '$Spec = Join-Path $Repo "installer\\yamabiko.spec"',
    )
    helper_contracts = (
        (
            "audit_bundle_inventory.py",
            ("--inventory $Inventory", "--dist $Dist"),
        ),
        (
            "bundle_provenance.py",
            (
                "resolve",
                "--inventory $Inventory",
                "--dist $Dist",
                "--policy-config $ProvenanceConfig",
                "--site-packages $SitePackages",
                "--output $ProvenanceReport",
            ),
        ),
        (
            "collect_licenses.py",
            (
                "--dist $Dist",
                "--inventory $Inventory",
                "--provenance-report $ProvenanceReport",
                "--override-policy $LicenseOverrides",
                "--site-packages $SitePackages",
            ),
        ),
        (
            "audit_license_closure.py",
            (
                "--dist $Dist",
                "--inventory $Inventory",
                "--license-index $LicenseIndex",
                "--provenance-policy-config $ProvenanceConfig",
                "--license-override-policy $LicenseOverrides",
                "--site-packages $SitePackages",
                "--output $ClosureAudit",
            ),
        ),
    )

    for script_path in script_paths:
        source = script_path.read_text(encoding="utf-8")
        commands = _active_powershell_commands(source)
        assert commands[:3] == ["param(", '[string]$Venv = ".venv"', ")"], (
            f'{script_path.name} must open with param([string]$Venv = ".venv"): '
            f"{commands[:3]}"
        )

        for assignment in exact_variables:
            assert commands.count(assignment) == 1, (
                f"{script_path.name} must actively define exactly once: {assignment}"
            )

        assert not any(
            command.casefold().startswith(
                ("python ", "pyinstaller ", "& python ", "& pyinstaller ")
            )
            for command in commands
        ), f"{script_path.name} contains a PATH-selected Python command"

        pip_commands = [
            command for command in commands
            if command.startswith("$FrozenLines =")
            and "pip freeze" in command.casefold()
        ]
        if script_path.name == "build.ps1":
            assert pip_commands == [
                "$FrozenLines = & $Python -m pip freeze --exclude-editable"
            ], f"build.ps1 must run pip freeze through the explicit venv: {pip_commands}"
            pip_index = commands.index(pip_commands[0])
            assert pip_index + 1 < len(commands) and commands[pip_index + 1].startswith(
                "if ($LASTEXITCODE -ne 0)"
            ), "build.ps1 must guard pip freeze immediately"
        else:
            assert not pip_commands, (
                f"{script_path.name} unexpectedly invokes pip freeze: {pip_commands}"
            )

        pyinstaller_commands = [
            command for command in commands
            if "-m pyinstaller" in command.casefold()
            or command.casefold().startswith("pyinstaller ")
        ]
        assert len(pyinstaller_commands) == 1, (
            f"{script_path.name} must invoke PyInstaller exactly once through $Python"
        )
        pyinstaller_command = pyinstaller_commands[0]
        assert pyinstaller_command.startswith("& $Python -m PyInstaller"), (
            f"{script_path.name} PyInstaller must use the explicit venv: "
            f"{pyinstaller_command}"
        )
        for argument in (
            "$Spec",
            "--noconfirm",
            '--workpath (Join-Path $Repo "build")',
            '--distpath (Join-Path $Repo "dist")',
        ):
            assert argument in pyinstaller_command, (
                f"{script_path.name} PyInstaller invocation is missing {argument}"
            )

        pyinstaller_index = commands.index(pyinstaller_command)
        assert pyinstaller_index + 1 < len(commands) and commands[
            pyinstaller_index + 1
        ].startswith("if ($LASTEXITCODE -ne 0)"), (
            f"{script_path.name} must guard PyInstaller immediately"
        )

        active_helper_commands = [
            command for command in commands
            if 'scripts\\' in command and ".py" in command
        ]
        assert len(active_helper_commands) == len(helper_contracts), (
            f"{script_path.name} has an unexpected Python helper set: "
            f"{active_helper_commands}"
        )

        helper_indices = []
        for helper_name, required_arguments in helper_contracts:
            matching = [
                command for command in active_helper_commands
                if helper_name in command
            ]
            assert len(matching) == 1, (
                f"{script_path.name} must invoke {helper_name} exactly once"
            )
            assert matching[0].startswith("& $Python "), (
                f"{script_path.name} must invoke {helper_name} through $Python"
            )
            for argument in required_arguments:
                assert argument in matching[0], (
                    f"{script_path.name} {helper_name} invocation is missing {argument}"
                )
            helper_index = commands.index(matching[0])
            helper_indices.append(helper_index)
            assert helper_index + 1 < len(commands) and commands[
                helper_index + 1
            ].startswith("if ($LASTEXITCODE -ne 0)"), (
                f"{script_path.name} must guard {helper_name} immediately"
            )

        command_indices = [pyinstaller_index, *helper_indices]
        assert command_indices == sorted(command_indices), (
            f"{script_path.name} release helpers are not in closure order"
        )
        copy_indices = [
            index
            for index, command in enumerate(commands)
            if command.startswith("Copy-Item ")
        ]
        assert copy_indices and max(copy_indices) < helper_indices[0], (
            f"{script_path.name} must finish every VC/llama copy before "
            "deriving inventory and license closure"
        )


# ---------------------------------------------------------------------------
# Section 7: check_pyinstaller_spec — lazy package import closure
# ---------------------------------------------------------------------------

def test_current_spec_declares_every_required_hiddenimport():
    """The real spec includes every package named by the preflight gate.

    Seeded defect: remove the explicit ``"faster_whisper"`` hiddenimport
    from installer/yamabiko.spec; section 7 reports it missing.
    """
    assert "faster_whisper" in preflight.SPEC_REQUIRED_HIDDENIMPORTS
    sec = preflight.check_pyinstaller_spec()
    assert not sec.failed(), sec.lines


def _complete_hiddenimports_assignment() -> str:
    literals = ", ".join(
        repr(name) for name in preflight.SPEC_REQUIRED_HIDDENIMPORTS
    )
    return f"hiddenimports = [{literals}]\n"


def _write_complete_spec(
    fake_repo: Path,
    *,
    before_analysis: str = "",
    after_analysis: str = "",
) -> None:
    (fake_repo / "installer" / "yamabiko.spec").write_text(
        _complete_hiddenimports_assignment()
        + before_analysis
        + "a = Analysis(hiddenimports=hiddenimports)\n"
        + after_analysis,
        encoding="utf-8",
    )


def _assert_spec_gate_fails_with(*expected: str) -> None:
    sec = preflight.check_pyinstaller_spec()
    assert sec.failed(), sec.lines
    assert any(
        status == "FAIL" and any(word in message for word in expected)
        for status, message in sec.lines
    ), sec.lines


@pytest.mark.parametrize(
    "consumer",
    [
        (
            "decoy = Analysis(hiddenimports=hiddenimports)\n"
            "a = Analysis(hiddenimports=[])\n"
        ),
        "if False:\n    a = Analysis(hiddenimports=hiddenimports)\n",
        (
            "def build():\n"
            "    return Analysis(hiddenimports=hiddenimports)\n"
        ),
        "a = PyInstaller.Analysis(hiddenimports=hiddenimports)\n",
        (
            "Factory = Analysis\n"
            "a = Analysis(hiddenimports=hiddenimports)\n"
            "decoy = Factory(hiddenimports=[])\n"
        ),
        (
            "Analysis = lambda **kwargs: object()\n"
            "a = Analysis(hiddenimports=hiddenimports)\n"
        ),
    ],
    ids=[
        "decoy-good-live-empty",
        "conditional-dead",
        "nested-call",
        "attribute-call",
        "analysis-alias",
        "analysis-rebind",
    ],
)
def test_spec_gate_rejects_noncanonical_analysis_consumers(fake_repo, consumer):
    """Only the executed module-level ``a = Analysis(...)`` may consume it.

    Seeded defects: add a good decoy beside a live empty consumer, move the
    consumer under a dead conditional, or call an attribute named Analysis.
    Each shape gave the old name-use counter one apparently valid consumer
    without proving what PyInstaller actually executes.
    """
    (fake_repo / "installer" / "yamabiko.spec").write_text(
        _complete_hiddenimports_assignment() + consumer,
        encoding="utf-8",
    )

    _assert_spec_gate_fails_with("Analysis")


@pytest.mark.parametrize(
    "escape",
    [
        'globals()["hiddenimports"].clear()',
        'locals()["hiddenimports"] = []',
        'vars()["hiddenimports"].clear()',
        'setattr(module, "hiddenimports", [])',
        'getattr(module, "hiddenimports").clear()',
        'module.__dict__["hiddenimports"] = []',
        'sys.modules[__name__].hiddenimports.clear()',
        'module.a = Factory(hiddenimports=[])',
        (
            "class RuntimeMutation:\n"
            "    sys.modules[__name__].hiddenimports.clear()"
        ),
        (
            "[sys.modules[__name__].hiddenimports.clear() "
            "for _ in (None,)]"
        ),
        'setattr(module, "Analysis", Factory)',
        'vars()["a"] = Factory(hiddenimports=[])',
        'exec("hiddenimports.clear()")',
        'eval("hiddenimports.clear()")',
        'namespace = globals\nnamespace()["hiddenimports"].clear()',
        'runner = exec\nrunner("hiddenimports.clear()")',
        '__builtins__["exec"]("hiddenimports.clear()")',
        '__builtins__["eval"]("hiddenimports.clear()")',
        "sys._getframe().f_globals.clear()",
    ],
    ids=[
        "globals",
        "locals",
        "vars",
        "setattr-hiddenimports",
        "getattr-hiddenimports",
        "module-dict-hiddenimports",
        "module-attribute-hiddenimports",
        "module-attribute-analysis-result",
        "class-body-module-attribute",
        "comprehension-body-module-attribute",
        "setattr-analysis",
        "vars-analysis-result",
        "exec",
        "eval",
        "globals-alias",
        "exec-alias",
        "builtins-exec-subscript",
        "builtins-eval-subscript",
        "frame-globals",
    ],
)
def test_spec_gate_rejects_reflective_or_dynamic_hiddenimports_escape(
    fake_repo,
    escape,
):
    """String-based access cannot bypass the audited-name proof.

    Seeded defects: mutate through ``globals``/``locals`` or dynamically
    execute source after the literal assignment. None creates an
    ``ast.Name('hiddenimports')`` use for the old scan to see.
    """
    _write_complete_spec(fake_repo, before_analysis=f"{escape}\n")

    _assert_spec_gate_fails_with("reflective", "dynamic", "hiddenimports")


@pytest.mark.parametrize(
    "exact_name_use",
    [
        (
            "def normalize(hiddenimports):\n"
            "    def inner():\n"
            "        return list(hiddenimports)\n"
            "    return inner()\n"
        ),
        (
            "def helper():\n"
            "    hiddenimports = []\n"
        ),
        (
            "class HelperConfig:\n"
            "    hiddenimports = []\n"
        ),
        "shadowed = [None for hiddenimports in ([],)]\n",
        (
            "class HelperConfig:\n"
            "    hiddenimports.clear()\n"
            "    hiddenimports = []\n"
        ),
        "shadowed = [None for hiddenimports in hiddenimports]\n",
        (
            "class RuntimeMutation:\n"
            "    hiddenimports *= 0\n"
        ),
        (
            "def outer(hiddenimports):\n"
            "    class RuntimeMutation:\n"
            "        hiddenimports.clear()\n"
            "        hiddenimports = []\n"
            "outer([])\n"
        ),
        (
            "class RuntimeMutation:\n"
            "    global hiddenimports\n"
            "    hiddenimports = []\n"
        ),
        (
            "def helper(value=(hiddenimports := [])):\n"
            "    pass\n"
        ),
        "hiddenimports.clear()\n",
        'hiddenimports[0] = "decoy"\n',
        "del hiddenimports[0]\n",
        "for hiddenimports in ([],):\n    pass\n",
        "from replacement import hiddenimports\n",
        "import replacement as hiddenimports\n",
        "def hiddenimports():\n    pass\n",
        "class hiddenimports:\n    pass\n",
        (
            "try:\n"
            "    raise RuntimeError()\n"
            "except RuntimeError as hiddenimports:\n"
            "    pass\n"
        ),
        (
            "match object():\n"
            "    case hiddenimports:\n"
            "        pass\n"
        ),
    ],
    ids=[
        "nested-parameter-and-closure",
        "function-local-store",
        "class-store",
        "comprehension-target",
        "class-load-before-store",
        "comprehension-enclosing-load",
        "class-augassign",
        "outer-parameter-shadowing-class",
        "class-global-store",
        "function-default-walrus",
        "method-call",
        "subscript-set",
        "subscript-delete",
        "loop-target",
        "from-import-binding",
        "import-alias-binding",
        "function-name-binding",
        "class-name-binding",
        "except-binding",
        "match-capture-binding",
    ],
)
def test_spec_gate_rejects_every_other_exact_hiddenimports_use_or_binding(
    fake_repo,
    exact_name_use,
):
    """The release-contract name is reserved everywhere except its two uses.

    The gate deliberately fails closed without a custom scope resolver: even
    a runtime-harmless nested local named ``hiddenimports`` is forbidden. This
    keeps the proof auditable while covering every AST Name context and the
    binders Python stores as text rather than ``ast.Name`` nodes.
    """
    _write_complete_spec(fake_repo, before_analysis=exact_name_use)

    _assert_spec_gate_fails_with("hiddenimports")


@pytest.mark.parametrize(
    "invalid_spec_fragment",
    [
        (
            "class RuntimeMutation:\n"
            "    [(hiddenimports := []) for _ in [0]]\n"
        ),
        (
            "def helper():\n"
            "    [(hiddenimports := []) for hiddenimports in [0]]\n"
        ),
    ],
    ids=["class-comprehension-walrus", "iteration-variable-walrus"],
)
def test_spec_gate_rejects_ast_parseable_semantic_syntax_errors(
    fake_repo,
    invalid_spec_fragment,
):
    """The spec must compile, not merely parse into an AST.

    Python defers these assignment-expression restrictions until ``compile``;
    neither form can execute as a PyInstaller spec even though ``ast.parse``
    accepts it and the structural membership proof otherwise false-greens.
    """
    _write_complete_spec(fake_repo, before_analysis=invalid_spec_fragment)

    _assert_spec_gate_fails_with("SyntaxError")


@pytest.mark.parametrize(
    "namespace_escape",
    [
        "from injected import *\n",
        "sys._getframe().f_globals.update(hiddenimports=[])\n",
        "helper(hiddenimports=[])\n",
        "sys._getframe().f_locals.clear()\n",
    ],
    ids=["wildcard", "frame-keyword", "keyword-only", "frame-locals"],
)
def test_spec_gate_rejects_direct_namespace_escape(
    fake_repo,
    namespace_escape,
):
    """Direct namespace spellings can overwrite contract-critical names."""
    _write_complete_spec(fake_repo, before_analysis=namespace_escape)

    _assert_spec_gate_fails_with("wildcard", "import", "keyword", "namespace")


@pytest.mark.parametrize(
    "analysis_result_rebind",
    [
        (
            "Factory = type(a)\n"
            "a = Factory(hiddenimports=[])\n"
        ),
        (
            "class RuntimeMutation:\n"
            "    global a\n"
            "    a = Factory(hiddenimports=[])\n"
        ),
        (
            "def helper(value=(a := Factory(hiddenimports=[]))):\n"
            "    pass\n"
        ),
        (
            "@((a := deco) and deco)\n"
            "def helper():\n"
            "    pass\n"
        ),
        "[(a := None) for _ in [0]]\n",
        (
            "class RuntimeBase((a := object)):\n"
            "    pass\n"
        ),
        "def helper():\n    a = None\n",
        "def helper(a):\n    return a\n",
        "class Helper:\n    a = None\n",
        "from replacement import a\n",
        "def a():\n    pass\n",
        "del a\n",
    ],
    ids=[
        "direct-rebind",
        "class-global-store",
        "function-default-walrus",
        "decorator-walrus",
        "comprehension-walrus",
        "class-base-walrus",
        "function-local-store",
        "function-parameter-binding",
        "class-store",
        "import-binding",
        "function-name-binding",
        "delete",
    ],
)
def test_spec_gate_rejects_every_other_analysis_result_store_or_binding(
    fake_repo,
    analysis_result_rebind,
):
    """The canonical Analysis target is the only allowed ``a`` binding.

    Loads remain available to downstream PYZ/EXE construction, but any other
    Store, Del, or text-stored binder fails closed regardless of nested scope.
    """
    _write_complete_spec(fake_repo, after_analysis=analysis_result_rebind)

    sec = preflight.check_pyinstaller_spec()

    assert sec.failed(), sec.lines
    assert any(
        status == "FAIL"
        and "a" in message
        and ("store" in message or "bind" in message)
        for status, message in sec.lines
    ), sec.lines


def test_spec_gate_allows_analysis_result_loads(fake_repo):
    """Downstream bundle construction may read the canonical ``a`` result."""
    _write_complete_spec(
        fake_repo,
        after_analysis="pyz = PYZ(a.pure)\nexe = EXE(a.scripts)\n",
    )

    sec = preflight.check_pyinstaller_spec()

    assert not sec.failed(), sec.lines


def test_spec_gate_rejects_missing_faster_whisper_hiddenimport(fake_repo):
    """The gate itself must require the lazy parent package.

    faster-whisper's data and dist-info can exist while its Python package
    code is absent from the frozen PYZ. This fixture includes every other
    required import, so it fails only if the parent-package guard exists.
    """
    other_required = [
        name
        for name in preflight.SPEC_REQUIRED_HIDDENIMPORTS
        if name != "faster_whisper"
    ]
    literals = ", ".join(repr(name) for name in other_required)
    (fake_repo / "installer" / "yamabiko.spec").write_text(
        "from PyInstaller.utils.hooks import collect_data_files, copy_metadata\n"
        'datas = collect_data_files("faster_whisper")\n'
        'datas += copy_metadata("faster-whisper")\n'
        f"hiddenimports = [{literals}]\n"
        "a = Analysis(hiddenimports=hiddenimports)\n",
        encoding="utf-8",
    )

    sec = preflight.check_pyinstaller_spec()

    assert sec.failed(), sec.lines
    assert any(
        status == "FAIL" and "faster_whisper" in message
        for status, message in sec.lines
    ), sec.lines


def test_spec_gate_rejects_duplicate_hiddenimports_assignments(fake_repo):
    """Two assignments make the final membership ambiguous to the gate.

    Seeded defect: accept the first/last assignment while ignoring the other;
    a later spec edit can replace the checked list after preflight approves it.
    """
    literals = ", ".join(repr(name) for name in preflight.SPEC_REQUIRED_HIDDENIMPORTS)
    (fake_repo / "installer" / "yamabiko.spec").write_text(
        f"hiddenimports = [{literals}]\n"
        f"hiddenimports = [{literals}]\n",
        encoding="utf-8",
    )

    sec = preflight.check_pyinstaller_spec()

    assert sec.failed(), sec.lines
    assert any(
        status == "FAIL" and "hiddenimports" in message
        and ("duplicate" in message or "multiple" in message)
        for status, message in sec.lines
    ), sec.lines


def test_spec_gate_rejects_computed_hiddenimports_assignment(fake_repo):
    """Required names must be direct strings, not an unverifiable builder.

    Seeded defect: a comprehension happens to quote every required name but
    filters one at runtime; a whole-file string search incorrectly approves it.
    """
    literals = ", ".join(repr(name) for name in preflight.SPEC_REQUIRED_HIDDENIMPORTS)
    (fake_repo / "installer" / "yamabiko.spec").write_text(
        f"hiddenimports = [name for name in ({literals}) if should_collect(name)]\n",
        encoding="utf-8",
    )

    sec = preflight.check_pyinstaller_spec()

    assert sec.failed(), sec.lines
    assert any(
        status == "FAIL" and "hiddenimports" in message
        and "direct string" in message
        for status, message in sec.lines
    ), sec.lines


def test_spec_gate_rejects_missing_hiddenimports_assignment_with_decoys(fake_repo):
    """Quoted names elsewhere cannot replace the audited assignment.

    Seeded defect: delete ``hiddenimports = [...]`` while leaving hook calls
    and comments that quote every required name; a whole-file search passes.
    """
    literals = ", ".join(repr(name) for name in preflight.SPEC_REQUIRED_HIDDENIMPORTS)
    (fake_repo / "installer" / "yamabiko.spec").write_text(
        f"decoy_hook_arguments = [{literals}]\n",
        encoding="utf-8",
    )

    sec = preflight.check_pyinstaller_spec()

    assert sec.failed(), sec.lines
    assert any(
        status == "FAIL" and "hiddenimports assignment" in message
        for status, message in sec.lines
    ), sec.lines


def test_spec_gate_rejects_non_string_direct_list_elements(fake_repo):
    """Only starred helper collections may be non-literal list elements.

    Seeded defect: append a runtime helper call to an otherwise complete
    literal list; membership is no longer fully auditable before executing the
    spec, so the gate must fail closed.
    """
    literals = ", ".join(repr(name) for name in preflight.SPEC_REQUIRED_HIDDENIMPORTS)
    (fake_repo / "installer" / "yamabiko.spec").write_text(
        f"hiddenimports = [{literals}, dynamic_hiddenimport()]\n",
        encoding="utf-8",
    )

    sec = preflight.check_pyinstaller_spec()

    assert sec.failed(), sec.lines
    assert any(
        status == "FAIL" and "direct string" in message
        for status, message in sec.lines
    ), sec.lines


def test_spec_gate_rejects_hiddenimports_list_not_passed_to_analysis(fake_repo):
    """A complete but unused list is not the bundle's actual membership.

    Seeded defect: pass an empty literal to ``Analysis`` while leaving the
    audited assignment intact; checking only that assignment false-greens.
    """
    literals = ", ".join(repr(name) for name in preflight.SPEC_REQUIRED_HIDDENIMPORTS)
    (fake_repo / "installer" / "yamabiko.spec").write_text(
        f"hiddenimports = [{literals}]\n"
        "a = Analysis(hiddenimports=[])\n",
        encoding="utf-8",
    )

    sec = preflight.check_pyinstaller_spec()

    assert sec.failed(), sec.lines
    assert any(
        status == "FAIL" and "Analysis" in message
        and "exactly once" in message
        for status, message in sec.lines
    ), sec.lines


def test_spec_gate_allows_starred_helpers_with_direct_required_strings(fake_repo):
    """The current spec's helper expansion is allowed beside direct pins.

    Seeded defect: reject every non-string element, including ``*helpers``;
    this healthy shape fails even though all release-critical names remain
    directly auditable.
    """
    literals = ", ".join(repr(name) for name in preflight.SPEC_REQUIRED_HIDDENIMPORTS)
    (fake_repo / "installer" / "yamabiko.spec").write_text(
        'helper_hiddenimports = collect_submodules("PySide6")\n'
        f"hiddenimports = [*helper_hiddenimports, {literals}]\n"
        "a = Analysis(hiddenimports=hiddenimports)\n",
        encoding="utf-8",
    )

    sec = preflight.check_pyinstaller_spec()

    assert not sec.failed(), sec.lines


# ---------------------------------------------------------------------------
# Section 3: check_version_consistency — binary VERSIONINFO quad guard
# ---------------------------------------------------------------------------

# .format()-substituted; the doubled ${{...}} braces render as the NSIS
# define reference ${APP_VERSION_QUAD}.
_NSI_TEMPLATE = """\
!define APP_VERSION "{app_version}"
!define APP_VERSION_QUAD "{quad}"
VIProductVersion "${{APP_VERSION_QUAD}}"
VIFileVersion "${{APP_VERSION_QUAD}}"
VIAddVersionKey "ProductVersion" "${{APP_VERSION}}"
VIAddVersionKey "FileVersion" "${{APP_VERSION}}"
"""

_VERINFO_TEMPLATE = """\
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={filevers},
    prodvers={prodvers},
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [
          StringStruct('FileVersion', '{string_version}'),
          StringStruct('ProductVersion', '{string_version}'),
        ]
      )
    ]),
  ]
)
"""


def _write_version_fixture(fake_repo, *, app_version="0.2.0-beta",
                           quad="0.2.0.0", filevers="(0, 2, 0, 0)",
                           prodvers=None, string_version=None,
                           nsi_src=None):
    """All five version-bearing files, mutually consistent by default.

    Each test seeds exactly one drift via the kwargs so the FAIL it
    asserts on is attributable to that drift alone."""
    prodvers = filevers if prodvers is None else prodvers
    string_version = app_version if string_version is None else string_version
    (fake_repo / "yamabiko" / "__init__.py").write_text(
        f'__version__ = "{app_version}"\n', encoding="utf-8")
    (fake_repo / "CHANGELOG.md").write_text(
        f"## [Unreleased]\n\n## [{app_version}] - 2026-01-01\n",
        encoding="utf-8")
    (fake_repo / "installer" / "README.md").write_text(
        f"Build produces yamabiko-{app_version}-setup.exe\n",
        encoding="utf-8")
    if nsi_src is None:
        nsi_src = _NSI_TEMPLATE.format(app_version=app_version, quad=quad)
    (fake_repo / "installer" / "yamabiko.nsi").write_text(
        nsi_src, encoding="utf-8")
    (fake_repo / "installer" / "version_info.txt").write_text(
        _VERINFO_TEMPLATE.format(filevers=filevers, prodvers=prodvers,
                                 string_version=string_version),
        encoding="utf-8")


def test_version_consistency_passes_against_current_repo():
    """The repo's version-bearing files (yamabiko/__init__.py,
    CHANGELOG.md, installer/yamabiko.nsi APP_VERSION + APP_VERSION_QUAD
    + VIProductVersion/VIFileVersion, installer/README.md,
    installer/version_info.txt tuples + strings) are mutually consistent
    right now, and the extended section 3 accepts them.

    Seeded defect: bump APP_VERSION in the repo .nsi without touching
    APP_VERSION_QUAD -> this fails (drilled 2026-07-01, see PR).
    """
    sec = preflight.check_version_consistency()
    assert not sec.failed(), sec.lines


def test_version_consistency_fixture_reports_both_string_and_quad_pass(fake_repo):
    """A fully-consistent fixture earns BOTH green lines: the legacy
    string agreement and the binary VERSIONINFO quad check. The second
    assertion is the guard's existence proof: if the quad sub-check is
    dropped from check_version_consistency, no PASS line mentions
    VERSIONINFO and this goes red.

    Seeded defect: delete the _binary_versioninfo_lines call from
    check_version_consistency -> this fails.
    """
    _write_version_fixture(fake_repo)
    sec = preflight.check_version_consistency()
    assert not sec.failed(), sec.lines
    pass_msgs = [m for s, m in sec.lines if s == "PASS"]
    assert any("agree" in m for m in pass_msgs), sec.lines
    assert any("VERSIONINFO" in m for m in pass_msgs), sec.lines


def test_version_consistency_fails_when_quad_and_tuples_stale_after_string_bump(fake_repo):
    """THE drift this guard exists for: every string version was bumped
    to 0.2.0-beta (so the legacy string check agrees and stays green)
    but the .nsi quad and the version_info.txt tuples still carry
    0.1.0.0 -- exactly what a "bump the version strings only" edit
    produces. The binary VERSIONINFO would ship stale.

    Seeded defect (Iron Law case): this fixture IS the one-line defect
    class; alternatively drop the quad comparison from
    check_version_consistency -> this goes red.
    """
    _write_version_fixture(fake_repo, quad="0.1.0.0", filevers="(0, 1, 0, 0)")
    sec = preflight.check_version_consistency()
    assert sec.failed(), sec.lines
    fail_msgs = [m for s, m in sec.lines if s == "FAIL"]
    assert any("APP_VERSION_QUAD" in m for m in fail_msgs), fail_msgs
    assert any("filevers" in m for m in fail_msgs), fail_msgs
    assert any("prodvers" in m for m in fail_msgs), fail_msgs


def test_version_consistency_fails_when_only_verinfo_tuples_stale(fake_repo):
    """The .nsi quad was bumped but installer/version_info.txt kept the
    old filevers/prodvers tuples: yamabiko.exe (spec EXE(version=...))
    ships the stale binary version even though the installer's is right.
    The tuples are compared against the APP_VERSION-derived quad, not
    against each other, so this half fails on its own.

    Seeded defect: drop the filevers/prodvers tuple comparison ->
    this goes red.
    """
    _write_version_fixture(fake_repo, filevers="(0, 1, 0, 0)")
    sec = preflight.check_version_consistency()
    assert sec.failed(), sec.lines
    fail_msgs = [m for s, m in sec.lines if s == "FAIL"]
    assert any("filevers" in m for m in fail_msgs), fail_msgs
    assert not any("APP_VERSION_QUAD" in m for m in fail_msgs), fail_msgs


def test_version_consistency_fails_when_verinfo_string_fields_stale(fake_repo):
    """version_info.txt's StringStruct FileVersion/ProductVersion must
    equal APP_VERSION. The legacy string check never read this file, so
    a bump that missed it shipped a yamabiko.exe whose Properties ->
    Details showed the previous release's version string.

    Seeded defect: drop the StringStruct comparison -> this goes red.
    """
    _write_version_fixture(fake_repo, string_version="0.1.0-beta")
    sec = preflight.check_version_consistency()
    assert sec.failed(), sec.lines
    fail_msgs = [m for s, m in sec.lines if s == "FAIL"]
    assert any("FileVersion" in m for m in fail_msgs), fail_msgs
    assert any("ProductVersion" in m for m in fail_msgs), fail_msgs


def test_version_consistency_fails_when_quad_define_missing(fake_repo):
    """An .nsi without APP_VERSION_QUAD (e.g. someone reverts to
    hardcoded VIProductVersion literals) must FAIL even when the
    literals happen to be numerically right today: the single-source
    structure is what keeps the NEXT bump from drifting.

    Seeded defect: make the missing-define branch report PASS (or skip
    the define lookup entirely) -> this goes red.
    """
    nsi_src = (
        '!define APP_VERSION "0.2.0-beta"\n'
        'VIProductVersion "0.2.0.0"\n'
        'VIFileVersion "0.2.0.0"\n'
    )
    _write_version_fixture(fake_repo, nsi_src=nsi_src)
    sec = preflight.check_version_consistency()
    assert sec.failed(), sec.lines
    fail_msgs = [m for s, m in sec.lines if s == "FAIL"]
    assert any("APP_VERSION_QUAD" in m for m in fail_msgs), fail_msgs


def test_version_consistency_fails_on_stale_hardcoded_vi_literal(fake_repo):
    """A correct APP_VERSION_QUAD define does not excuse a VI key that
    bypasses it with a stale literal: VIProductVersion/VIFileVersion are
    checked by RESOLVED VALUE (define reference or literal), so a
    half-reverted .nsi still fails.

    Seeded defect: compare only the define and skip the VI keys ->
    this goes red.
    """
    nsi_src = (
        '!define APP_VERSION "0.2.0-beta"\n'
        '!define APP_VERSION_QUAD "0.2.0.0"\n'
        'VIProductVersion "0.1.0.0"\n'
        'VIFileVersion "${APP_VERSION_QUAD}"\n'
    )
    _write_version_fixture(fake_repo, nsi_src=nsi_src)
    sec = preflight.check_version_consistency()
    assert sec.failed(), sec.lines
    fail_msgs = [m for s, m in sec.lines if s == "FAIL"]
    assert any("VIProductVersion" in m for m in fail_msgs), fail_msgs


# ---------------------------------------------------------------------------
# Section 10: check_config_key_consumers
# ---------------------------------------------------------------------------

def test_config_key_consumers_all_keys_referenced(fake_repo):
    (fake_repo / "yamabiko" / "default_config.yaml").write_text(
        "audio:\n  device: default\n  sample_rate: 16000\n",
        encoding="utf-8",
    )
    # Consumer file with both keys as string literals.
    (fake_repo / "yamabiko" / "main.py").write_text(
        'cfg["audio"]["device"]\ncfg["audio"]["sample_rate"]\n',
        encoding="utf-8",
    )
    sec = preflight.check_config_key_consumers()
    assert not sec.failed(), sec.lines
    assert any("all 2 leaf key(s)" in msg for _, msg in sec.lines)


def test_config_key_consumers_orphan_key_fails(fake_repo):
    (fake_repo / "yamabiko" / "default_config.yaml").write_text(
        "audio:\n  device: default\n  orphan_key: unused\n",
        encoding="utf-8",
    )
    # Only one of the two keys has a consumer reference.
    (fake_repo / "yamabiko" / "main.py").write_text(
        'cfg["audio"]["device"]\n',
        encoding="utf-8",
    )
    sec = preflight.check_config_key_consumers()
    assert sec.failed()
    fail_msgs = [msg for status, msg in sec.lines if status == "FAIL"]
    assert any("orphan_key" in m for m in fail_msgs)
    assert any("audio.orphan_key" in m for m in fail_msgs)


def test_config_key_consumers_collision_leaf_orphan_fails(fake_repo):
    """Two keys share the leaf name ``threshold``; the consumer reads
    only ``vad.threshold``. A bare leaf-literal grep is satisfied by
    the OTHER key's consumer, so the orphan ships silently (the
    lifecycle.grace_s.overlay false-pass class). The check must
    require the immediate parent key near the leaf literal for
    collision-prone leaves and FAIL on the unanchored one.

    Seeded defect: drop the parent-anchor branch from
    check_config_key_consumers -> this test goes red.
    """
    (fake_repo / "yamabiko" / "default_config.yaml").write_text(
        "vad:\n  threshold: 0.5\n"
        "asr:\n  faster_whisper:\n    threshold: 0.01\n",
        encoding="utf-8",
    )
    (fake_repo / "yamabiko" / "main.py").write_text(
        'cfg["vad"]["threshold"]\n',
        encoding="utf-8",
    )
    sec = preflight.check_config_key_consumers()
    assert sec.failed(), sec.lines
    fail_msgs = [msg for status, msg in sec.lines if status == "FAIL"]
    assert any("asr.faster_whisper.threshold" in m for m in fail_msgs), fail_msgs
    # The anchored consumer must NOT be flagged.
    assert not any("vad.threshold" in m for m in fail_msgs), fail_msgs


def test_config_key_consumers_collision_leaf_with_anchored_consumers_passes(fake_repo):
    """Both colliding keys have parent-anchored consumers -> green.

    Seeded defect: tighten the anchor window to 0 chars (parent must be
    immediately adjacent with no ``][`` between) -> this goes red.
    """
    (fake_repo / "yamabiko" / "default_config.yaml").write_text(
        "vad:\n  threshold: 0.5\n"
        "asr:\n  faster_whisper:\n    threshold: 0.01\n",
        encoding="utf-8",
    )
    (fake_repo / "yamabiko" / "main.py").write_text(
        'cfg["vad"]["threshold"]\n'
        'cfg["asr"]["faster_whisper"].get("threshold")\n',
        encoding="utf-8",
    )
    sec = preflight.check_config_key_consumers()
    assert not sec.failed(), sec.lines


def test_config_key_consumers_collision_with_container_name_requires_anchor(fake_repo):
    """A leaf colliding with a CONTAINER key name (not another leaf) is
    also collision-prone: ``overlay`` here is both a container and a
    leaf (lifecycle.grace_s.overlay can be falsely satisfied by an
    unrelated "overlay" role string in shutdown_event.py).

    Seeded defect: build the collision set from leaf names only ->
    ``overlay`` stops being collision-prone, the unanchored hit
    satisfies the check, and this test goes red.
    """
    (fake_repo / "yamabiko" / "default_config.yaml").write_text(
        "overlay:\n  opacity: 0.9\n"
        "lifecycle:\n  grace_s:\n    overlay: 3\n",
        encoding="utf-8",
    )
    # "overlay" appears as an unrelated role string; "opacity" is read
    # properly; the grace_s consumer is MISSING.
    (fake_repo / "yamabiko" / "main.py").write_text(
        'ROLE = "overlay"\ncfg["overlay"]["opacity"]\n',
        encoding="utf-8",
    )
    sec = preflight.check_config_key_consumers()
    assert sec.failed(), sec.lines
    fail_msgs = [msg for status, msg in sec.lines if status == "FAIL"]
    assert any("lifecycle.grace_s.overlay" in m for m in fail_msgs), fail_msgs


def test_config_key_consumers_allowlist_excludes_container_key(fake_repo):
    # `style` is in the allowlist — its children don't need direct
    # literal references (they're iterated via the parent dict).
    (fake_repo / "yamabiko" / "default_config.yaml").write_text(
        "overlay:\n  style:\n    jp_font: Yu Gothic UI\n",
        encoding="utf-8",
    )
    (fake_repo / "yamabiko" / "main.py").write_text(
        'cfg["overlay"]["style"]["jp_font"]\n',
        encoding="utf-8",
    )
    sec = preflight.check_config_key_consumers()
    assert not sec.failed(), sec.lines


# ---------------------------------------------------------------------------
# Section 11: check_bundle_module_launchers
# ---------------------------------------------------------------------------

def _write_launcher_fixture(fake_repo, *, spec_hiddenimports: str,
                             main_src: str = "", watchdog_src: str = "",
                             build_src: str = ""):
    (fake_repo / "installer" / "yamabiko.spec").write_text(
        f"hiddenimports = [{spec_hiddenimports}]\n", encoding="utf-8",
    )
    (fake_repo / "installer" / "build.ps1").write_text(build_src, encoding="utf-8")
    (fake_repo / "yamabiko" / "main.py").write_text(main_src, encoding="utf-8")
    (fake_repo / "yamabiko" / "watchdog.py").write_text(watchdog_src, encoding="utf-8")


def test_bundle_launchers_overlay_with_dispatch_passes(fake_repo):
    _write_launcher_fixture(
        fake_repo,
        spec_hiddenimports='"yamabiko.overlay_app.__main__"',
        main_src='if sys.argv[1] == "--run-overlay": ...',
        watchdog_src="",  # dispatch can be in either file
        build_src="",
    )
    sec = preflight.check_bundle_module_launchers()
    assert not sec.failed(), sec.lines


def test_bundle_launchers_main_without_dispatch_fails(fake_repo):
    _write_launcher_fixture(
        fake_repo,
        spec_hiddenimports='"yamabiko.overlay_app.__main__"',
        main_src="pass",  # no --run-overlay dispatch
        watchdog_src="pass",
        build_src="",
    )
    sec = preflight.check_bundle_module_launchers()
    assert sec.failed()
    fail_msgs = [msg for status, msg in sec.lines if status == "FAIL"]
    assert any("no matching argv dispatch" in m for m in fail_msgs)


def test_bundle_launchers_llama_bundle_without_spawn_fails(fake_repo):
    _write_launcher_fixture(
        fake_repo,
        # A valid __main__ + dispatch pair keeps the empty-spec drift
        # guard quiet so this test stays focused on the llama half.
        spec_hiddenimports='"yamabiko.overlay_app.__main__"',
        main_src='if sys.argv[1] == "--run-overlay": ...',
        watchdog_src="pass",  # no _build_llama_cmd
        build_src='Copy-Item llama-server.exe',
    )
    sec = preflight.check_bundle_module_launchers()
    assert sec.failed()
    fail_msgs = [msg for status, msg in sec.lines if status == "FAIL"]
    assert any("bundled but never launched" in m for m in fail_msgs)


def test_bundle_launchers_llama_paired_passes(fake_repo):
    _write_launcher_fixture(
        fake_repo,
        spec_hiddenimports='"yamabiko.overlay_app.__main__"',
        main_src='if sys.argv[1] == "--run-overlay": ...',
        watchdog_src="def _build_llama_cmd(): pass",
        build_src='Copy-Item llama-server.exe',
    )
    sec = preflight.check_bundle_module_launchers()
    assert not sec.failed(), sec.lines


def test_bundle_launchers_no_llama_refs_is_skip_not_fail(fake_repo):
    # The llama half has nothing to check -> SKIP, not FAIL. The spec
    # carries a valid __main__ + dispatch so the section as a whole
    # stays green (an EMPTY spec is now a FAIL by design).
    _write_launcher_fixture(
        fake_repo,
        spec_hiddenimports='"yamabiko.overlay_app.__main__"',
        main_src='if sys.argv[1] == "--run-overlay": ...',
        watchdog_src="pass",
        build_src="",
    )
    sec = preflight.check_bundle_module_launchers()
    assert not sec.failed()
    assert any(status == "SKIP" for status, _ in sec.lines)


def test_bundle_launchers_empty_spec_hiddenimports_fails(fake_repo):
    """The bundled-but-never-launched guard rests on a regex finding
    ``yamabiko.*.__main__`` literals in the spec. If a spec refactor
    (f-string / loop-built hiddenimports) retires those literals, the
    regex finds nothing and the guard silently stops guarding while the
    section stays green. Zero matches must FAIL loudly.

    Seeded defect: drop the ``if not main_modules: sec.add("FAIL", ...)``
    guard from check_bundle_module_launchers -> this test goes red.
    """
    _write_launcher_fixture(
        fake_repo,
        spec_hiddenimports='',  # no __main__ literals at all
        main_src="pass",
        watchdog_src="def _build_llama_cmd(): pass",
        build_src='Copy-Item llama-server.exe',  # llama half green
    )
    sec = preflight.check_bundle_module_launchers()
    assert sec.failed(), sec.lines
    fail_msgs = [msg for status, msg in sec.lines if status == "FAIL"]
    assert any("drifted" in m and "cannot verify launcher coupling" in m for m in fail_msgs), fail_msgs


def test_bundle_launchers_detects_dotted_submodule_main(fake_repo):
    """``yamabiko.a.b.__main__`` literals must be detected too — with
    the old ``(\\w+)`` capture the regex skips dotted module paths,
    which (combined with the empty-spec drift guard) would FAIL a
    healthy spec.

    Seeded defect: revert the capture group to ``(\\w+)`` -> the
    literal is invisible, the empty-spec guard fires, and this goes red.
    """
    _write_launcher_fixture(
        fake_repo,
        spec_hiddenimports='"yamabiko.tools.probe.__main__"',
        main_src='if sys.argv[1] == "--run-tools.probe": ...',
        watchdog_src="",
        build_src="",
    )
    sec = preflight.check_bundle_module_launchers()
    assert not sec.failed(), sec.lines


# ---------------------------------------------------------------------------
# Section 12: check_changelog_migration_notes
# ---------------------------------------------------------------------------

def _init_git_with_tag(repo: Path, tag: str = "v0.0.1") -> None:
    """Initialise a git repo and tag the initial commit."""
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env={**env})
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env={**env})
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, env={**env})
    subprocess.run(["git", "tag", tag], cwd=repo, check=True, env={**env})


def test_migration_notes_skips_when_no_tags(fake_repo):
    (fake_repo / "yamabiko" / "default_config.yaml").write_text("k: v\n", encoding="utf-8")
    (fake_repo / "CHANGELOG.md").write_text("## [Unreleased]\n", encoding="utf-8")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q"], cwd=fake_repo, check=True, env={**env})
    # No tags created.
    sec = preflight.check_changelog_migration_notes()
    assert not sec.failed()
    assert any(status == "SKIP" for status, _ in sec.lines)


def test_migration_notes_passes_when_config_unchanged(fake_repo):
    (fake_repo / "yamabiko" / "default_config.yaml").write_text("k: v\n", encoding="utf-8")
    (fake_repo / "CHANGELOG.md").write_text("## [Unreleased]\n", encoding="utf-8")
    _init_git_with_tag(fake_repo)
    sec = preflight.check_changelog_migration_notes()
    assert not sec.failed()
    assert any("unchanged" in msg for _, msg in sec.lines)


def test_migration_notes_warns_when_config_changed_without_migration(fake_repo):
    (fake_repo / "yamabiko" / "default_config.yaml").write_text("k: v\n", encoding="utf-8")
    (fake_repo / "CHANGELOG.md").write_text("## [Unreleased]\n\n- misc\n", encoding="utf-8")
    _init_git_with_tag(fake_repo)
    # Modify the config after tagging and commit.
    (fake_repo / "yamabiko" / "default_config.yaml").write_text("k: v2\n", encoding="utf-8")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "commit", "-aqm", "bump"], cwd=fake_repo, check=True, env={**env})
    sec = preflight.check_changelog_migration_notes()
    # WARN, not FAIL — preflight stays green but message surfaces.
    assert not sec.failed()
    assert any(status == "WARN" for status, _ in sec.lines)


# ---------------------------------------------------------------------------
# Section 8: check_real_path_pytest — pytest-timeout fail-fast
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_real_path_section_fails_fast_when_pytest_timeout_missing(monkeypatch):
    """Section 8 passes ``--timeout=180``; on a fresh box without
    pytest-timeout, pytest exits 4 ("unrecognized arguments") and the
    classifier reports "real-path suite FAILED" — indistinguishable
    from a genuine test failure. The section must probe the plugin in
    the TARGET interpreter first and FAIL with a message naming
    pytest-timeout, without ever invoking the real suite.

    Seeded defect: drop the plugin pre-probe from check_real_path_pytest
    -> the FAIL message no longer names pytest-timeout and the suite
    invocation fires anyway -> both assertions below go red.
    """
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return _FakeProc(
            returncode=1,
            stderr="ModuleNotFoundError: No module named 'pytest_timeout'",
        )

    monkeypatch.setattr(preflight.subprocess, "run", _fake_run)
    sec = preflight.check_real_path_pytest()
    assert sec.failed()
    fail_msgs = [m for s, m in sec.lines if s == "FAIL"]
    assert any("pytest-timeout" in m for m in fail_msgs), (
        f"section 8 FAIL message does not name the missing plugin: {fail_msgs}"
    )
    # The probe must short-circuit: exactly one subprocess (the import
    # probe), and the real-tier pytest run must never have been issued.
    assert len(calls) == 1, (
        f"expected only the plugin import probe, got {len(calls)} calls: {calls}"
    )
    assert calls[0][1:] == ["-c", "import pytest_timeout"], (
        f"first subprocess is not the pytest_timeout import probe: {calls[0]}"
    )


def test_real_path_section_runs_suite_when_pytest_timeout_present(monkeypatch):
    """Happy path: with the plugin importable (probe rc=0), the section
    proceeds to the lightweight real invocation and classifies its result as
    before. The pre-build section must exclude tests that require an already
    built candidate; the pre-probe must not block a healthy env.

    Seeded defect: make the pre-probe unconditionally FAIL (inverted
    rc check) -> the suite never runs and this goes red.
    """
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if list(cmd[1:3]) == ["-c", "import pytest_timeout"]:
            return _FakeProc(returncode=0)
        return _FakeProc(returncode=0, stdout="5 passed, 2 skipped in 12.3s\n")

    monkeypatch.setattr(preflight.subprocess, "run", _fake_run)
    sec = preflight.check_real_path_pytest()
    assert not sec.failed(), sec.lines
    assert len(calls) == 2, f"expected probe + suite run, got: {calls}"
    assert "pytest" in calls[1], (
        f"second subprocess is not a pytest suite run: {calls[1]}"
    )
    marker_index = max(index for index, arg in enumerate(calls[1]) if arg == "-m")
    assert calls[1][marker_index + 1] == "real and not frozen_artifact", (
        "preflight section 8 runs before the frozen candidate is required; "
        f"its selector must exclude frozen_artifact, got {calls[1]}"
    )


def test_real_path_section_runs_the_selected_venvs_python(monkeypatch, tmp_path):
    """--venv picks the interpreter section 8 runs the real tier in, so a
    release preflight tests the release venv's packages, not the dev venv's
    (a package only the dev venv has, such as Pillow, would hide a missing
    dependency).

    Seeded defect: keep ``ROOT / ".venv"`` in check_real_path_pytest -> the
    fixture has no .venv, so it falls back to sys.executable -> red.
    """
    python = tmp_path / ".venv-release" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if list(cmd[1:3]) == ["-c", "import pytest_timeout"]:
            return _FakeProc(returncode=0)
        return _FakeProc(returncode=0, stdout="5 passed in 1.0s\n")

    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(preflight.subprocess, "run", _fake_run)

    sec = preflight.check_real_path_pytest(venv=".venv-release")

    assert not sec.failed(), sec.lines
    assert [call[0] for call in calls] == [str(python), str(python)]


def test_real_path_section_fails_naming_an_explicit_venv_without_python(monkeypatch, tmp_path):
    """An explicit --venv whose interpreter is missing (a typo, a venv not
    created yet) fails section 8 and names the path it looked for. Only the
    default .venv falls back to preflight's own interpreter (a contributor
    box without one); an explicit venv that fell back would report a tier it
    never ran in that venv as passed.

    Seeded defect: fall back to sys.executable for every venv -> the tier
    runs in this interpreter and the section passes -> red.
    """
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if list(cmd[1:3]) == ["-c", "import pytest_timeout"]:
            return _FakeProc(returncode=0)
        return _FakeProc(returncode=0, stdout="5 passed in 1.0s\n")

    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(preflight.subprocess, "run", _fake_run)

    sec = preflight.check_real_path_pytest(venv=".venv-relase")

    fail_msgs = [m for s, m in sec.lines if s == "FAIL"]
    missing = tmp_path / ".venv-relase" / "Scripts" / "python.exe"
    assert any(str(missing) in m for m in fail_msgs), sec.lines
    assert calls == []


def test_bundle_audit_section_fails_naming_an_explicit_venv_without_python(
    monkeypatch, tmp_path,
):
    """Section 9's twin of the section 8 rule: with a built bundle, an
    explicit --venv whose interpreter is missing fails the DLL audit, names
    the path it looked for and runs nothing.

    Seeded defect: drop section 9's missing-interpreter check -> the audit
    runs as ``[None, audit_bundle_dlls.py]`` (here the fake passes it) -> no
    FAIL -> red.
    """
    (tmp_path / "dist" / "yamabiko").mkdir(parents=True)
    calls: list[list] = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return _FakeProc(returncode=0, stdout="Native files: 1\n")

    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(preflight.subprocess, "run", _fake_run)

    sec = preflight.check_bundle_dll_audit(venv=".venv-relase")

    fail_msgs = [m for s, m in sec.lines if s == "FAIL"]
    missing = tmp_path / ".venv-relase" / "Scripts" / "python.exe"
    assert any(str(missing) in m for m in fail_msgs), sec.lines
    assert calls == []


# ---------------------------------------------------------------------------
# Section 13: check_no_subject_mocks_in_real_tier
# ---------------------------------------------------------------------------

def _make_test_file(fake_repo, name, body):
    (fake_repo / "tests").mkdir(exist_ok=True)
    (fake_repo / "tests" / name).write_text(body, encoding="utf-8")


def test_real_tier_mock_check_passes_when_no_mocks(fake_repo):
    _make_test_file(
        fake_repo, "test_real_foo.py",
        "import pytest\n@pytest.mark.real\ndef test_foo():\n    assert 1 == 1\n",
    )
    sec = preflight.check_no_subject_mocks_in_real_tier()
    assert not sec.failed(), sec.lines


def test_real_tier_mock_check_fails_on_magicmock_call(fake_repo):
    _make_test_file(
        fake_repo, "test_real_foo.py",
        "import pytest\nfrom unittest.mock import MagicMock\n"
        "@pytest.mark.real\ndef test_foo():\n    m = MagicMock()\n    assert m\n",
    )
    sec = preflight.check_no_subject_mocks_in_real_tier()
    assert sec.failed()
    fail_msgs = [msg for status, msg in sec.lines if status == "FAIL"]
    assert any("MagicMock()" in m for m in fail_msgs)


def test_real_tier_mock_check_ignores_docstring_mention(fake_repo):
    # The bare word MagicMock in a docstring is NOT a usage site.
    _make_test_file(
        fake_repo, "test_real_foo.py",
        'import pytest\n'
        '"""This file deliberately does NOT use MagicMock — see docs."""\n'
        '@pytest.mark.real\ndef test_foo():\n    assert 1 == 1\n',
    )
    sec = preflight.check_no_subject_mocks_in_real_tier()
    assert not sec.failed(), sec.lines


def test_real_tier_mock_check_ignores_unit_tier_files(fake_repo):
    # A file with BOTH a @pytest.mark.real test AND unit-tier tests
    # with mocks is acceptable as long as it's not named test_real_*
    # and doesn't have a module-level pytestmark = pytest.mark.real.
    _make_test_file(
        fake_repo, "test_foo.py",
        "import pytest\nfrom unittest.mock import MagicMock\n"
        "def test_unit():\n    m = MagicMock()\n    assert m\n"
        "@pytest.mark.real\ndef test_real():\n    assert 1 == 1\n",
    )
    sec = preflight.check_no_subject_mocks_in_real_tier()
    assert not sec.failed(), sec.lines


def test_real_tier_mock_check_fires_on_module_level_pytestmark(fake_repo):
    # A file with `pytestmark = pytest.mark.real` is fully real-tier
    # even if named test_foo.py — the blanket marker is binding.
    _make_test_file(
        fake_repo, "test_foo.py",
        "import pytest\nfrom unittest.mock import MagicMock\n"
        "pytestmark = pytest.mark.real\n"
        "def test_foo():\n    m = MagicMock()\n    assert m\n",
    )
    sec = preflight.check_no_subject_mocks_in_real_tier()
    assert sec.failed()


def test_migration_notes_passes_when_migration_section_present(fake_repo):
    (fake_repo / "yamabiko" / "default_config.yaml").write_text("k: v\n", encoding="utf-8")
    (fake_repo / "CHANGELOG.md").write_text(
        "## [Unreleased]\n\n### Migration\n\nRename `k` to `kk`.\n",
        encoding="utf-8",
    )
    _init_git_with_tag(fake_repo)
    (fake_repo / "yamabiko" / "default_config.yaml").write_text("kk: v\n", encoding="utf-8")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "commit", "-aqm", "rename"], cwd=fake_repo, check=True, env={**env})
    sec = preflight.check_changelog_migration_notes()
    assert not sec.failed()
    assert any("Migration subsection" in msg for _, msg in sec.lines)


# ---------------------------------------------------------------------------
# Section 14: check_llama_bundle_closure
# ---------------------------------------------------------------------------

def _write_llama_manifest(fake_repo, entries):
    (fake_repo / "installer" / "llama-manifest.txt").write_text(
        "# manifest\n" + "\n".join(entries) + "\n", encoding="utf-8",
    )


# Plausible PE bytes: MZ magic + padding past the 4 KB stub floor.
# The closure check rejects name-only matches whose content cannot be
# a real binary, so fixture files must look like PEs.
_FAKE_PE = b"MZ" + b"\x00" * 8192


def _make_llama_bundle(fake_repo, files):
    llama = fake_repo / "dist" / "yamabiko" / "_internal" / "llama"
    llama.mkdir(parents=True, exist_ok=True)
    for f in files:
        (llama / f).write_bytes(_FAKE_PE)
    return llama


def test_llama_bundle_closure_passes_when_all_present(fake_repo):
    _write_llama_manifest(fake_repo, ["llama-server.exe", "ggml.dll"])
    _make_llama_bundle(fake_repo, ["llama-server.exe", "ggml.dll"])
    sec = preflight.check_llama_bundle_closure()
    assert not sec.failed(), sec.lines


def test_llama_bundle_closure_fails_on_missing_dll(fake_repo):
    _write_llama_manifest(fake_repo, ["llama-server.exe", "ggml-cuda.dll"])
    _make_llama_bundle(fake_repo, ["llama-server.exe"])  # ggml-cuda.dll absent
    sec = preflight.check_llama_bundle_closure()
    assert sec.failed()
    assert any("ggml-cuda.dll" in m for _, m in sec.lines)


def test_llama_bundle_closure_skips_without_bundle_in_dev(fake_repo):
    _write_llama_manifest(fake_repo, ["llama-server.exe"])
    sec = preflight.check_llama_bundle_closure(release=False)
    assert not sec.failed()
    assert any(s == "SKIP" for s, _ in sec.lines)


def test_llama_bundle_closure_fails_without_bundle_under_release(fake_repo):
    _write_llama_manifest(fake_repo, ["llama-server.exe"])
    sec = preflight.check_llama_bundle_closure(release=True)
    assert sec.failed()


def test_llama_bundle_closure_fails_on_text_stub_binary(fake_repo):
    """ci-prepare-vendor-stubs.ps1 creates small text-file stubs with
    the exact manifest names; every downstream guard was existence-only,
    so a stub created locally (partial re-vendor) could ship in a real
    release installer with a dead translator. A manifest .exe/.dll whose
    content cannot be a PE image must FAIL the closure.

    Seeded defect: revert check_llama_bundle_closure to the name-only
    membership check."""
    _write_llama_manifest(fake_repo, ["llama-server.exe", "ggml.dll"])
    llama = _make_llama_bundle(fake_repo, ["ggml.dll"])
    (llama / "llama-server.exe").write_text(
        "CI stub (not a real binary)\n", encoding="utf-8",
    )
    sec = preflight.check_llama_bundle_closure()
    assert sec.failed()
    fail_msgs = [m for s, m in sec.lines if s == "FAIL"]
    assert any("llama-server.exe" in m and "stub" in m.lower() for m in fail_msgs)


def test_llama_bundle_closure_fails_on_large_non_pe_binary(fake_repo):
    """A binary-sized file without the MZ magic is still not loadable;
    the size floor alone must not green-light it. Seeded defect: drop
    the MZ-magic half of the stub check, keep the size floor."""
    _write_llama_manifest(fake_repo, ["ggml.dll"])
    llama = fake_repo / "dist" / "yamabiko" / "_internal" / "llama"
    llama.mkdir(parents=True)
    (llama / "ggml.dll").write_bytes(b"\x00" * 8192)  # big, but no MZ
    sec = preflight.check_llama_bundle_closure()
    assert sec.failed()


# ---------------------------------------------------------------------------
# scripts/mutmut-analyze.py — dynamic section bounds + accounting
# ---------------------------------------------------------------------------

def _load_mutmut_analyze():
    """Load scripts/mutmut-analyze.py (dash in name -> importlib)."""
    path = Path(preflight.__file__).resolve().parent / "mutmut-analyze.py"
    spec = importlib.util.spec_from_file_location("mutmut_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_cache(path: Path, files: dict[str, list[tuple[int, str, str]]]) -> None:
    """Build a minimal mutmut 2.x cache.

    ``files`` maps filename -> [(line_number_0based, line_text, status)],
    one mutant per entry.
    """
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE SourceFile (id INTEGER PRIMARY KEY, filename TEXT)")
    con.execute(
        "CREATE TABLE Line (id INTEGER PRIMARY KEY, sourcefile INTEGER,"
        " line_number INTEGER, line TEXT)"
    )
    con.execute("CREATE TABLE Mutant (id INTEGER PRIMARY KEY, line INTEGER, status TEXT)")
    fid = lid = mid = 0
    for fname, mutants in files.items():
        fid += 1
        con.execute("INSERT INTO SourceFile VALUES (?, ?)", (fid, fname))
        for line0, text, status in mutants:
            lid += 1
            con.execute("INSERT INTO Line VALUES (?, ?, ?, ?)", (lid, fid, line0, text))
            mid += 1
            con.execute("INSERT INTO Mutant VALUES (?, ?, ?)", (mid, lid, status))
    con.commit()
    con.close()


def _write_min_targets(root: Path) -> None:
    """The two target files the watchdog fixture doesn't cover."""
    (root / "yamabiko").mkdir(exist_ok=True)
    (root / "yamabiko" / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
    ov = root / "yamabiko" / "overlay_app"
    ov.mkdir(exist_ok=True)
    (ov / "__main__.py").write_text("def run_overlay():\n    pass\n", encoding="utf-8")


def test_mutmut_analyze_sections_derive_from_current_source():
    """Bounds come from the source text, not a hardcoded table.

    Seeded defect: classify against a stale hardcoded line table ->
    a def living past the old ceiling maps to an out-of-bounds bucket
    instead of its own name -> both assertions go red.
    """
    mod = _load_mutmut_analyze()
    pad = "# filler\n" * 1190  # well past the old hardcoded watchdog ceiling
    src = pad + (
        "def build_watchdog():\n"
        "    a = 1\n"
        "\n"
        "class WatchdogApp:\n"
        "    def handle_pause(self):\n"
        "        pass\n"
    )
    secs = mod.sections_for(src)
    def_line = src.splitlines().index("def build_watchdog():") + 1
    assert mod.section_at(secs, def_line + 1) == "build_watchdog"
    method_line = src.splitlines().index("    def handle_pause(self):") + 1
    assert mod.section_at(secs, method_line + 1) == "WatchdogApp.handle_pause"
    # No line in the file can land in an unreported bucket.
    all_secs = {mod.section_at(secs, n) for n in range(1, len(src.splitlines()) + 1)}
    assert not any(s.startswith("out-of-bounds") for s in all_secs), all_secs


def test_mutmut_analyze_composition_membership_is_by_owner_name():
    """Composition surface follows the owner's NAME across refactors.

    Seeded defect: drop ``WatchdogApp`` from COMPOSITION_OWNERS (or
    match on section string equality so class methods fall out) ->
    the method case goes red.
    """
    mod = _load_mutmut_analyze()
    assert mod.is_composition("build_watchdog")
    assert mod.is_composition("class WatchdogApp")
    assert mod.is_composition("WatchdogApp.handle_pause")
    assert not mod.is_composition("_build_llama_cmd")
    assert not mod.is_composition("(module level)")


def test_mutmut_analyze_reports_mutants_beyond_old_bounds_and_audits_totals(
    tmp_path, capsys,
):
    """A mutant at line ~1193 of watchdog.py (past the old hardcoded
    ceiling) must appear in a named section row AND in the accounting.

    Seeded defect: reintroduce a fixed section list in the report loop
    -> the ``zzz_new_helper`` row disappears, the audit counts fewer
    mutants than the cache holds, and main() exits non-zero -> all
    three assertions go red.
    """
    mod = _load_mutmut_analyze()
    pad = "# filler\n" * 1190
    src = pad + (
        "def build_watchdog():\n"
        "    a = 1\n"
        "\n"
        "def zzz_new_helper():\n"
        "    b = 2\n"
    )
    _write_min_targets(tmp_path)
    (tmp_path / "yamabiko" / "watchdog.py").write_text(src, encoding="utf-8")
    lines = src.splitlines()
    bw_body = lines.index("    a = 1")        # 0-based, mutmut convention
    helper_body = lines.index("    b = 2")
    cache = tmp_path / ".mutmut-cache"
    _make_cache(cache, {
        "yamabiko/watchdog.py": [
            (bw_body, "    a = 1", "ok_killed"),
            (helper_body, "    b = 2", "bad_survived"),
            (helper_body, "    b = 2", "untested"),
        ],
    })
    rc = mod.main(cache=str(cache), root=tmp_path)
    out = capsys.readouterr().out
    assert "build_watchdog" in out
    assert "zzz_new_helper" in out, out
    assert "Total mutants: 3" in out
    assert rc == 0, out


def test_mutmut_analyze_unknown_file_mutants_stay_in_the_accounting(
    tmp_path, capsys,
):
    """Mutants of files outside the target set must be surfaced as
    UNKNOWN-FILE rows, not silently dropped.

    Seeded defect: skip non-target files in the bucket loop -> the
    audit mismatch flips main() to exit 1 -> red.
    """
    mod = _load_mutmut_analyze()
    _write_min_targets(tmp_path)
    (tmp_path / "yamabiko" / "watchdog.py").write_text(
        "def build_watchdog():\n    a = 1\n", encoding="utf-8",
    )
    cache = tmp_path / ".mutmut-cache"
    _make_cache(cache, {
        "yamabiko/watchdog.py": [(1, "    a = 1", "ok_killed")],
        "yamabiko/stray.py": [(0, "x = 9", "bad_survived")],
    })
    rc = mod.main(cache=str(cache), root=tmp_path)
    out = capsys.readouterr().out
    assert "UNKNOWN-FILE" in out
    assert "Total mutants: 2" in out
    assert rc == 0, out


def test_llama_bundle_closure_fails_when_bundle_exists_but_llama_subtree_missing(fake_repo):
    """The silent-ship case: a real bundle was built, but the entire
    _internal/llama subtree is absent (llama never vendored in). This must
    FAIL -- even in dev -- not be mistaken for a 'no bundle' dev SKIP.

    Seeded defect: treat a missing _internal/llama under an existing bundle
    as the no-bundle SKIP -> this fails.
    """
    _write_llama_manifest(fake_repo, ["llama-server.exe"])
    # Bundle exists, but no _internal/llama subtree.
    (fake_repo / "dist" / "yamabiko" / "_internal").mkdir(parents=True)
    sec = preflight.check_llama_bundle_closure(release=False)
    assert sec.failed()
