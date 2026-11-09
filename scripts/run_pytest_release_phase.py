"""Run one fail-closed pytest phase from ``release-gates.json``.

Each phase is collected independently before execution.  The release-only
pytest plugin then compares the executed node IDs with that collection and
turns skips, xfails, incomplete execution, or collection drift into failures.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_PLUGIN = "scripts.pytest_release_gate"
_PHASE_ID = re.compile(r"[a-z0-9][a-z0-9-]*\Z")


def _load_phase(root: Path, phase_id: str) -> tuple[list[str], int]:
    manifest_path = root / "scripts" / "release-gates.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("release-gates.json schema_version must be 1")
    if not _PHASE_ID.fullmatch(phase_id):
        raise ValueError(f"invalid release phase ID: {phase_id!r}")

    phases = payload.get("phases")
    if not isinstance(phases, list):
        raise ValueError("release-gates.json phases must be a list")
    if not all(isinstance(phase, dict) for phase in phases):
        raise ValueError("release-gates.json phases must contain objects")
    matches = [phase for phase in phases if phase.get("id") == phase_id]
    if len(matches) != 1:
        available = ", ".join(
            str(phase.get("id")) for phase in phases if isinstance(phase, dict)
        )
        raise ValueError(
            f"unknown or duplicate release phase {phase_id!r}; "
            f"available phases: {available}"
        )

    phase = matches[0]
    args = phase.get("args")
    timeout = phase.get("timeout_seconds")
    if not isinstance(args, list) or not all(
        isinstance(arg, str) for arg in args
    ):
        raise ValueError(f"release phase {phase_id!r} args must be strings")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError(
            f"release phase {phase_id!r} timeout_seconds must be positive"
        )
    return args, timeout


def _subprocess_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(root) if not existing else str(root) + os.pathsep + existing
    )
    return env


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
            timeout=15,
        )
    else:
        process.kill()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


def _run(command: list[str], root: Path, timeout: int, label: str) -> int:
    print(f"\n== {label} ==", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    process = subprocess.Popen(command, cwd=root, env=_subprocess_env(root))
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"ERROR: {label} timed out after {timeout} seconds", file=sys.stderr)
        _terminate_process_tree(process)
        return 124


def run_phase(root: Path, phase_id: str, python_executable: Path) -> int:
    """Collect and execute one manifest phase, returning its gate status."""
    root = root.resolve()
    args, timeout = _load_phase(root, phase_id)
    evidence_dir = root / "build" / "release-gates"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    expected = evidence_dir / f"{phase_id}-expected.json"
    evidence = evidence_dir / f"{phase_id}-evidence.json"
    expected.unlink(missing_ok=True)
    evidence.unlink(missing_ok=True)

    base = [str(python_executable), "-m", "pytest", "-q", "-p", _PLUGIN]
    collect_command = [
        *base,
        "-q",
        "-q",
        "--collect-only",
        "--yamabiko-collection-output",
        str(expected),
        *args,
    ]
    collection_status = _run(
        collect_command,
        root,
        min(timeout, 300),
        f"Collect release phase: {phase_id}",
    )
    if collection_status != 0:
        return collection_status

    execute_command = [
        *base,
        "--yamabiko-release-phase",
        phase_id,
        "--yamabiko-expected-collection",
        str(expected),
        "--yamabiko-phase-evidence",
        str(evidence),
        *args,
    ]
    return _run(
        execute_command,
        root,
        timeout,
        f"Execute release phase: {phase_id}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", help="phase ID from scripts/release-gates.json")
    args = parser.parse_args(argv)
    try:
        return run_phase(ROOT, args.phase, Path(sys.executable))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
