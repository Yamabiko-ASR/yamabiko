"""Release-only pytest collection and execution accounting.

Load explicitly with ``-p scripts.pytest_release_gate``.  The plugin remains
inert for contributor runs unless a Yamabiko collection or release option is
present.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


@dataclass
class _ReportOutcome:
    outcome: str
    wasxfail: bool


@dataclass
class _GateState:
    phase_id: str | None
    collection_output: Path | None
    expected_collection: Path | None
    phase_evidence: Path | None
    collected_nodeids: list[str] = field(default_factory=list)
    reports: dict[str, dict[str, _ReportOutcome]] = field(default_factory=dict)
    collection_skips: list[str] = field(default_factory=list)
    build_binding: dict[str, Any] = field(default_factory=dict)


_STATE: _GateState | None = None
# Collect reports arrive before ``pytest_collection_finish`` creates the
# gate state, so skipped collectors are buffered here per session.
_COLLECTION_SKIPS: list[str] = []


def _git_output(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _build_binding(root: Path) -> dict[str, Any]:
    """Bind phase evidence to the commit, tree state and frozen build.

    Captured before any test runs, so files a test leaves behind cannot mark
    the tree dirty and the record describes the code that was exercised.
    """
    head = (_git_output(root, "rev-parse", "HEAD") or "").strip() or None
    status = _git_output(root, "status", "--porcelain")
    inventory = root / "build" / "yamabiko" / "bundle-inventory.json"
    try:
        inventory_sha256: str | None = hashlib.sha256(
            inventory.read_bytes()
        ).hexdigest()
    except OSError:
        inventory_sha256 = None
    return {
        "git_head": head,
        "git_dirty": bool(status.strip()) if status is not None else None,
        "bundle_inventory_sha256": inventory_sha256,
    }


def _normalized_nodeids(nodeids: list[str]) -> list[str]:
    return sorted(set(nodeids))


def _nodeids_sha256(nodeids: list[str]) -> str:
    canonical = "\n".join(_normalized_nodeids(nodeids)) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    serialized = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _collection_payload(nodeids: list[str]) -> dict[str, Any]:
    normalized = _normalized_nodeids(nodeids)
    return {
        "schema_version": 1,
        "nodeids": normalized,
        "nodeids_sha256": _nodeids_sha256(normalized),
    }


def _load_expected_collection(
    path: Path | None,
) -> tuple[bool, list[str], str | None, str | None]:
    if path is None:
        return False, [], None, "expected collection path was not supplied"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, [], None, f"cannot load expected collection: {exc}"
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return False, [], None, "expected collection schema_version must be 1"
    nodeids = payload.get("nodeids")
    recorded_hash = payload.get("nodeids_sha256")
    if not isinstance(nodeids, list) or not all(
        isinstance(nodeid, str) for nodeid in nodeids
    ):
        return False, [], None, "expected collection nodeids must be strings"
    normalized = _normalized_nodeids(nodeids)
    if nodeids != normalized:
        return False, [], None, "expected collection nodeids are not canonical"
    computed_hash = _nodeids_sha256(normalized)
    if recorded_hash != computed_hash:
        return False, [], None, "expected collection hash is invalid"
    return True, normalized, computed_hash, None


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("yamabiko release gate")
    group.addoption(
        "--yamabiko-release-phase",
        metavar="PHASE_ID",
        help="Enable fail-closed execution accounting for a release phase.",
    )
    group.addoption(
        "--yamabiko-collection-output",
        metavar="JSON_PATH",
        help="Atomically write canonical collected node IDs to JSON.",
    )
    group.addoption(
        "--yamabiko-expected-collection",
        metavar="JSON_PATH",
        help="Canonical collection JSON expected for release execution.",
    )
    group.addoption(
        "--yamabiko-phase-evidence",
        metavar="JSON_PATH",
        help="Atomic JSON output for release execution evidence.",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    del session
    _COLLECTION_SKIPS.clear()


def pytest_collectreport(report: pytest.CollectReport) -> None:
    """Record collectors skipped while collecting, such as a module-level
    ``pytest.importorskip``. Their tests never become items, so neither the
    expected collection nor the executed set can see them."""
    if report.skipped:
        _COLLECTION_SKIPS.append(report.nodeid)


def pytest_collection_finish(session: pytest.Session) -> None:
    global _STATE

    config = session.config
    phase_id = config.getoption("yamabiko_release_phase")
    collection_output_text = config.getoption("yamabiko_collection_output")
    if phase_id is None and collection_output_text is None:
        _STATE = None
        return

    _STATE = _GateState(
        phase_id=phase_id,
        collection_output=(
            Path(collection_output_text) if collection_output_text else None
        ),
        expected_collection=(
            Path(config.getoption("yamabiko_expected_collection"))
            if config.getoption("yamabiko_expected_collection")
            else None
        ),
        phase_evidence=(
            Path(config.getoption("yamabiko_phase_evidence"))
            if config.getoption("yamabiko_phase_evidence")
            else None
        ),
        collected_nodeids=_normalized_nodeids(
            [item.nodeid for item in session.items]
        ),
        collection_skips=sorted(set(_COLLECTION_SKIPS)),
        build_binding=(
            _build_binding(config.rootpath) if phase_id is not None else {}
        ),
    )
    if _STATE.collection_output is not None:
        _atomic_write_json(
            _STATE.collection_output,
            _collection_payload(_STATE.collected_nodeids),
        )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _STATE is None or _STATE.phase_id is None:
        return
    if report.when not in {"setup", "call", "teardown"}:
        return
    _STATE.reports.setdefault(report.nodeid, {})[report.when] = _ReportOutcome(
        outcome=report.outcome,
        wasxfail=hasattr(report, "wasxfail"),
    )


def _execution_evidence(
    state: _GateState,
    original_exit_status: int,
) -> tuple[dict[str, Any], list[str]]:
    expected_valid, expected_nodeids, expected_hash, expected_error = (
        _load_expected_collection(state.expected_collection)
    )
    collected = state.collected_nodeids
    collected_set = set(collected)
    collection_matches = expected_valid and expected_nodeids == collected

    executed_nodeids: list[str] = []
    missing_nodeids: list[str] = []
    incomplete_nodeids: list[str] = []
    unexpected_nodeids = sorted(set(state.reports) - collected_set)
    passed_nodeids: list[str] = []
    failed_nodeids: list[str] = []
    error_nodeids: list[str] = []
    skipped_nodeids: list[str] = []
    xfailed_nodeids: list[str] = []
    xpassed_nodeids: list[str] = []

    for nodeid in sorted(collected_set | set(state.reports)):
        reports = state.reports.get(nodeid, {})
        if not reports:
            missing_nodeids.append(nodeid)
            continue

        complete = {"setup", "call", "teardown"}.issubset(reports)
        if complete:
            executed_nodeids.append(nodeid)

        setup_or_teardown_error = any(
            reports.get(when) is not None
            and reports[when].outcome == "failed"
            for when in ("setup", "teardown")
        )
        xfail_reports = [item for item in reports.values() if item.wasxfail]
        if xfail_reports:
            if any(item.outcome == "skipped" for item in xfail_reports):
                xfailed_nodeids.append(nodeid)
            else:
                xpassed_nodeids.append(nodeid)
        elif any(item.outcome == "skipped" for item in reports.values()):
            skipped_nodeids.append(nodeid)

        if setup_or_teardown_error:
            error_nodeids.append(nodeid)
        elif (
            reports.get("call") is not None
            and reports["call"].outcome == "failed"
            and not xfail_reports
        ):
            failed_nodeids.append(nodeid)
        elif (
            complete
            and not xfail_reports
            and nodeid not in skipped_nodeids
            and all(item.outcome == "passed" for item in reports.values())
        ):
            passed_nodeids.append(nodeid)

        if not complete and not setup_or_teardown_error and not any(
            item.outcome == "skipped" for item in reports.values()
        ):
            incomplete_nodeids.append(nodeid)

    violations: list[str] = []
    if not expected_valid:
        violations.append("expected_collection_invalid")
    elif not collection_matches:
        violations.append("collection_mismatch")
    violations.extend(
        f"collection-time skip: {nodeid}" for nodeid in state.collection_skips
    )
    if missing_nodeids:
        violations.append("missing_execution_reports")
    if incomplete_nodeids or unexpected_nodeids:
        violations.append("incomplete_execution_accounting")
    for name, nodeids in (
        ("failed", failed_nodeids),
        ("errors", error_nodeids),
        ("skipped", skipped_nodeids),
        ("xfailed", xfailed_nodeids),
        ("xpassed", xpassed_nodeids),
    ):
        if nodeids:
            violations.append(name)

    final_exit_status = (
        original_exit_status
        if original_exit_status != 0
        else (1 if violations else 0)
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "phase_id": state.phase_id,
        "expected_collection_valid": expected_valid,
        "expected_collection_error": expected_error,
        "collection_matches_expected": collection_matches,
        "expected_collected": len(expected_nodeids) if expected_valid else 0,
        "collected": len(collected),
        "executed": len(executed_nodeids),
        "passed": len(passed_nodeids),
        "failed": len(failed_nodeids),
        "errors": len(error_nodeids),
        "skipped": len(skipped_nodeids),
        "xfailed": len(xfailed_nodeids),
        "xpassed": len(xpassed_nodeids),
        "original_pytest_exit_status": original_exit_status,
        "final_exit_status": final_exit_status,
        "collected_nodeids": collected,
        "collected_nodeids_sha256": _nodeids_sha256(collected),
        "expected_nodeids_sha256": expected_hash,
        "executed_nodeids": executed_nodeids,
        "executed_nodeids_sha256": _nodeids_sha256(executed_nodeids),
        "missing_nodeids": missing_nodeids,
        "incomplete_nodeids": incomplete_nodeids,
        "unexpected_nodeids": unexpected_nodeids,
        "passed_nodeids": passed_nodeids,
        "failed_nodeids": failed_nodeids,
        "error_nodeids": error_nodeids,
        "skipped_nodeids": skipped_nodeids,
        "xfailed_nodeids": xfailed_nodeids,
        "xpassed_nodeids": xpassed_nodeids,
        "collection_skips": state.collection_skips,
        "git_head": state.build_binding.get("git_head"),
        "git_dirty": state.build_binding.get("git_dirty"),
        "bundle_inventory_sha256": state.build_binding.get(
            "bundle_inventory_sha256"
        ),
        "violations": violations,
    }
    return payload, violations


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del exitstatus
    if _STATE is None or _STATE.phase_id is None:
        return

    original_exit_status = int(session.exitstatus)
    payload, violations = _execution_evidence(_STATE, original_exit_status)
    if _STATE.phase_evidence is None:
        if original_exit_status == 0:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        return

    _atomic_write_json(_STATE.phase_evidence, payload)
    if original_exit_status == 0 and violations:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
