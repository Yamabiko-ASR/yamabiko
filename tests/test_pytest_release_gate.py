"""Behavioral subprocess tests for the release-only pytest gate plugin."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON = Path(sys.executable)
_PLUGIN = "scripts.pytest_release_gate"


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(_REPO_ROOT)
        if not old_pythonpath
        else str(_REPO_ROOT) + os.pathsep + old_pythonpath
    )
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return env


def _run_temp_pytest(suite: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(_PYTHON),
            "-m",
            "pytest",
            "-q",
            "--rootdir",
            str(suite),
            "-p",
            _PLUGIN,
            *args,
        ],
        cwd=suite,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )


def _run_project_collection(
    output: Path, *selector: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(_PYTHON),
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            _PLUGIN,
            "--yamabiko-collection-output",
            str(output),
            *selector,
        ],
        cwd=_REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )


def _write_test(suite: Path, source: str) -> None:
    suite.mkdir()
    (suite / "test_sample.py").write_text(source, encoding="utf-8")


def _nodeids_sha256(nodeids: list[str]) -> str:
    canonical = "\n".join(sorted(set(nodeids))) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_expected(path: Path, nodeids: list[str]) -> None:
    normalized = sorted(set(nodeids))
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "nodeids": normalized,
                "nodeids_sha256": _nodeids_sha256(normalized),
            }
        ),
        encoding="utf-8",
    )


def _release_args(expected: Path, evidence: Path) -> tuple[str, ...]:
    return (
        "--yamabiko-release-phase",
        "unit",
        "--yamabiko-expected-collection",
        str(expected),
        "--yamabiko-phase-evidence",
        str(evidence),
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contributor_mode_preserves_skip_exit_zero(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    _write_test(
        suite,
        "import pytest\n\ndef test_needs_resource():\n"
        "    pytest.skip('missing prerequisite')\n",
    )

    result = _run_temp_pytest(suite)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 skipped" in result.stdout


def test_release_mode_turns_skip_into_exit_one_and_records_it(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    _write_test(
        suite,
        "import pytest\n\ndef test_needs_resource():\n"
        "    pytest.skip('missing prerequisite')\n",
    )
    expected = tmp_path / "expected.json"
    evidence = tmp_path / "evidence.json"
    _write_expected(expected, ["test_sample.py::test_needs_resource"])

    result = _run_temp_pytest(suite, *_release_args(expected, evidence))

    assert result.returncode == 1, result.stdout + result.stderr
    payload = _read_json(evidence)
    assert payload["phase_id"] == "unit"
    assert payload["original_pytest_exit_status"] == 0
    assert payload["final_exit_status"] == 1
    assert payload["collected"] == 1
    assert payload["executed"] == 1
    assert payload["passed"] == 0
    assert payload["skipped"] == 1
    assert payload["skipped_nodeids"] == [
        "test_sample.py::test_needs_resource"
    ]


def test_release_mode_rejects_xfail_and_xpass(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    _write_test(
        suite,
        "import pytest\n\n"
        "@pytest.mark.xfail(reason='known')\n"
        "def test_expected_failure():\n    assert False\n\n"
        "@pytest.mark.xfail(reason='known')\n"
        "def test_unexpected_pass():\n    assert True\n",
    )
    expected = tmp_path / "expected.json"
    evidence = tmp_path / "evidence.json"
    _write_expected(
        expected,
        [
            "test_sample.py::test_expected_failure",
            "test_sample.py::test_unexpected_pass",
        ],
    )

    result = _run_temp_pytest(suite, *_release_args(expected, evidence))

    assert result.returncode == 1, result.stdout + result.stderr
    payload = _read_json(evidence)
    assert payload["original_pytest_exit_status"] == 0
    assert payload["xfailed"] == 1
    assert payload["xpassed"] == 1
    assert payload["executed"] == 2


def test_release_mode_rejects_collection_hash_mismatch(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    _write_test(suite, "def test_present():\n    pass\n")

    cases = {
        "mismatch": json.dumps(
            {
                "schema_version": 1,
                "nodeids": ["test_sample.py::test_different"],
                "nodeids_sha256": _nodeids_sha256(
                    ["test_sample.py::test_different"]
                ),
            }
        ),
        "malformed": "{not valid JSON",
        "missing": None,
    }
    for case, content in cases.items():
        expected = tmp_path / f"expected-{case}.json"
        evidence = tmp_path / f"evidence-{case}.json"
        if content is not None:
            expected.write_text(content, encoding="utf-8")

        result = _run_temp_pytest(suite, *_release_args(expected, evidence))

        assert result.returncode == 1, (
            case + "\n" + result.stdout + result.stderr
        )
        payload = _read_json(evidence)
        assert payload["final_exit_status"] == 1
        assert payload["collection_matches_expected"] is False
        if case == "mismatch":
            assert payload["expected_collection_valid"] is True
        else:
            assert payload["expected_collection_valid"] is False


def test_release_mode_rejects_incomplete_execution_accounting(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    _write_test(suite, "def test_never_run():\n    pass\n")
    (suite / "conftest.py").write_text(
        "def pytest_runtestloop(session):\n    return True\n",
        encoding="utf-8",
    )
    expected = tmp_path / "expected.json"
    evidence = tmp_path / "evidence.json"
    _write_expected(expected, ["test_sample.py::test_never_run"])

    result = _run_temp_pytest(suite, *_release_args(expected, evidence))

    assert result.returncode == 1, result.stdout + result.stderr
    payload = _read_json(evidence)
    assert payload["original_pytest_exit_status"] == 0
    assert payload["collected"] == 1
    assert payload["executed"] == 0
    assert payload["missing_nodeids"] == ["test_sample.py::test_never_run"]


def test_release_mode_records_setup_and_teardown_errors(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    _write_test(
        suite,
        "import pytest\n\n"
        "@pytest.fixture\n"
        "def broken_setup():\n    raise RuntimeError('setup boom')\n\n"
        "@pytest.fixture\n"
        "def broken_teardown():\n"
        "    yield\n"
        "    raise RuntimeError('teardown boom')\n\n"
        "def test_setup_error(broken_setup):\n    pass\n\n"
        "def test_teardown_error(broken_teardown):\n    pass\n",
    )
    expected = tmp_path / "expected.json"
    evidence = tmp_path / "evidence.json"
    _write_expected(
        expected,
        [
            "test_sample.py::test_setup_error",
            "test_sample.py::test_teardown_error",
        ],
    )

    result = _run_temp_pytest(suite, *_release_args(expected, evidence))

    assert result.returncode == 1, result.stdout + result.stderr
    payload = _read_json(evidence)
    assert payload["original_pytest_exit_status"] == 1
    assert payload["final_exit_status"] == 1
    assert payload["errors"] == 2
    assert payload["executed"] == 1
    assert payload["error_nodeids"] == [
        "test_sample.py::test_setup_error",
        "test_sample.py::test_teardown_error",
    ]


def test_release_mode_writes_evidence_when_the_session_fails(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    _write_test(suite, "def test_failure():\n    assert 1 == 2\n")
    expected = tmp_path / "expected.json"
    evidence = tmp_path / "evidence.json"
    _write_expected(expected, ["test_sample.py::test_failure"])

    result = _run_temp_pytest(suite, *_release_args(expected, evidence))

    assert result.returncode == 1, result.stdout + result.stderr
    payload = _read_json(evidence)
    assert payload["original_pytest_exit_status"] == 1
    assert payload["final_exit_status"] == 1
    assert payload["failed"] == 1
    assert payload["executed"] == 1
    assert payload["failed_nodeids"] == ["test_sample.py::test_failure"]


def test_release_mode_turns_module_level_importorskip_into_violation(
    tmp_path: Path,
) -> None:
    """A module skipped while it is collected never produces items, so the
    expected collection and the executed set both silently omit it.

    Seeded defect: drop the ``pytest_collectreport`` hook; the phase then
    exits 0 with the skipped module invisible in its evidence.
    """
    suite = tmp_path / "suite"
    _write_test(suite, "def test_present():\n    pass\n")
    (suite / "test_needs_missing_module.py").write_text(
        "import pytest\n\n"
        "pytest.importorskip('yamabiko_release_gate_absent_module')\n\n"
        "def test_never_collected():\n    pass\n",
        encoding="utf-8",
    )
    expected = tmp_path / "expected.json"
    evidence = tmp_path / "evidence.json"
    _write_expected(expected, ["test_sample.py::test_present"])

    result = _run_temp_pytest(suite, *_release_args(expected, evidence))

    assert result.returncode == 1, result.stdout + result.stderr
    payload = _read_json(evidence)
    assert payload["original_pytest_exit_status"] == 0
    assert payload["final_exit_status"] == 1
    assert payload["passed"] == 1
    assert payload["collection_skips"] == ["test_needs_missing_module.py"]
    assert payload["violations"] == [
        "collection-time skip: test_needs_missing_module.py"
    ]


def test_collection_only_mode_writes_canonical_nodeids_without_parsing_stdout(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    _write_test(
        suite,
        "def test_zeta():\n    pass\n\ndef test_alpha():\n    pass\n",
    )
    output = tmp_path / "collection.json"

    result = _run_temp_pytest(
        suite,
        "--collect-only",
        "--yamabiko-collection-output",
        str(output),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = _read_json(output)
    expected_nodeids = [
        "test_sample.py::test_alpha",
        "test_sample.py::test_zeta",
    ]
    assert payload == {
        "schema_version": 1,
        "nodeids": expected_nodeids,
        "nodeids_sha256": _nodeids_sha256(expected_nodeids),
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_frozen_e2e_markers_form_exact_runtime_and_build_partitions(
    tmp_path: Path,
) -> None:
    target = "tests/test_real_frozen_e2e.py"
    full_output = tmp_path / "full.json"
    external_output = tmp_path / "external.json"
    owned_output = tmp_path / "owned.json"
    artifact_output = tmp_path / "artifact.json"
    prebuild_output = tmp_path / "prebuild.json"

    full_result = _run_project_collection(full_output, target)
    external_result = _run_project_collection(
        external_output,
        "-m",
        "frozen_external_llama",
        target,
    )
    owned_result = _run_project_collection(
        owned_output,
        "-m",
        "frozen_owned_llama",
        target,
    )
    artifact_result = _run_project_collection(
        artifact_output,
        "-m",
        "frozen_artifact",
        target,
    )
    prebuild_result = _run_project_collection(
        prebuild_output,
        "-m",
        "not frozen_artifact",
        target,
    )
    for result in (
        full_result,
        external_result,
        owned_result,
        artifact_result,
    ):
        assert result.returncode == 0, result.stdout + result.stderr
    assert prebuild_result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, (
        prebuild_result.stdout + prebuild_result.stderr
    )

    full = set(_read_json(full_output)["nodeids"])
    external = set(_read_json(external_output)["nodeids"])
    owned = set(_read_json(owned_output)["nodeids"])
    artifact = set(_read_json(artifact_output)["nodeids"])
    prebuild = set(_read_json(prebuild_output)["nodeids"])
    expected_external = {
        f"{target}::test_frozen_exe_starts_without_errors_for_45s",
        f"{target}::test_frozen_exe_translates_sample_wav_and_broadcasts_subtitle[faster-subtitles-on]",
        f"{target}::test_frozen_exe_translates_sample_wav_and_broadcasts_subtitle[faster-subtitles-off]",
    }

    assert len(full) == 15
    assert external == expected_external
    assert len(owned) == 12
    assert external.isdisjoint(owned)
    assert external | owned == full
    assert artifact == full
    assert prebuild == set()


def test_release_phase_runner_distinguishes_complete_pass_from_skip(
    tmp_path: Path,
) -> None:
    """The operator-facing runner must execute the plugin, not merely pytest.

    A complete phase returns zero and records one pass. A phase containing a
    real pytest skip returns nonzero and records that skip as the violation.
    Removing the plugin or release options makes one side of this distinction
    fail.
    """
    runner_path = _REPO_ROOT / "scripts" / "run_pytest_release_phase.py"
    assert runner_path.is_file(), "release phase runner is missing"

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(
        _REPO_ROOT / "scripts" / "pytest_release_gate.py",
        scripts_dir / "pytest_release_gate.py",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        "import pytest\n\n"
        "def test_pass():\n    assert 2 + 2 == 4\n\n"
        "def test_skip():\n    pytest.skip('missing release prerequisite')\n",
        encoding="utf-8",
    )
    (scripts_dir / "release-gates.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phases": [
                    {
                        "id": "pass",
                        "timeout_seconds": 30,
                        "args": ["tests/test_sample.py::test_pass"],
                    },
                    {
                        "id": "skip",
                        "timeout_seconds": 30,
                        "args": ["tests/test_sample.py::test_skip"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    spec = importlib.util.spec_from_file_location(
        "_yamabiko_release_phase_runner_test", runner_path,
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    assert runner.run_phase(tmp_path, "pass", Path(sys.executable)) == 0
    pass_evidence = _read_json(
        tmp_path / "build" / "release-gates" / "pass-evidence.json"
    )
    assert pass_evidence["passed"] == 1
    assert pass_evidence["violations"] == []

    assert runner.run_phase(tmp_path, "skip", Path(sys.executable)) == 1
    skip_evidence = _read_json(
        tmp_path / "build" / "release-gates" / "skip-evidence.json"
    )
    assert skip_evidence["skipped"] == 1
    assert skip_evidence["violations"] == ["skipped"]
