"""Run-request selection of the system under test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.e0_evidence import runtime
from scripts.e0_evidence.replay import vad_assist_layout
from scripts.e0_evidence.runtime import RUN_REQUEST_SCHEMA_VERSION, run_request
from scripts.e0_evidence.wire import canonical_sha256
from tests.e0_evidence.test_score_runs import _verified_run
from yamabiko.vad import sound_tagger


_REPO = Path(__file__).resolve().parents[2]
_COHORT_ID = "96000000-0000-4000-8000-000000000003"
# Captured before any fixture can stub the check for other unit tests.
_REAL_VERIFY_RELEASE_LOCK = runtime._verify_release_lock


def test_declared_vad_artifact_is_frozen_into_run_start_and_cohort_and_mismatch_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onnxruntime

    fixture = _verified_run(tmp_path, monkeypatch)
    root = fixture.boundary.root
    (master_path,) = (root / "masters").glob("*/master.private.json")
    master = json.loads(master_path.read_text("utf-8"))
    (reference_dir,) = (root / "references").iterdir()
    model_dir = tmp_path / "ct2-snapshot"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"synthetic ct2 weights")
    config_path = root / "config.yaml"
    config_path.write_text(
        f"asr:\n  faster_whisper:\n    model: '{model_dir.as_posix()}'\n",
        "utf-8",
    )
    candidate = tmp_path / "candidate-silero.onnx"
    candidate.write_bytes(b"candidate silero graph")
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()

    def write_request(request_id: str, declared_sha256: str) -> Path:
        path = root / f"{request_id}.request.private.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": RUN_REQUEST_SCHEMA_VERSION,
                    "request_id": request_id,
                    "master_id": master["master_id"],
                    "reference_set_id": reference_dir.name,
                    "config_path": str(config_path),
                    "model_artifact_path": str(model_dir),
                    "vad_artifact": {
                        "path": str(candidate),
                        "sha256": declared_sha256,
                    },
                    "arms": ["p0"],
                    "arm_ids": {"p0": "96000000-0000-4000-8000-000000000002"},
                    "cohort_id": _COHORT_ID,
                    "cohort_fixture_ids": [master["fixture_id"]],
                    "oracle_components": [],
                }
            ),
            "utf-8",
        )
        return path

    sessions: list[str] = []

    def record_session(path: str, **_kwargs: object) -> None:
        # Stop the run once the declared graph reaches ONNX Runtime.
        sessions.append(path)
        raise RuntimeError("synthetic bytes are not an ONNX graph")

    monkeypatch.setattr(onnxruntime, "InferenceSession", record_session)

    final = run_request(
        write_request("96000000-0000-4000-8000-000000000001", candidate_sha256),
        fixture.boundary,
        repo=_REPO,
        argv=("run",),
    ).final

    start = json.loads((root / "runs" / final.run_id / "run-start.json").read_bytes())
    identity = {
        "sha256": candidate_sha256,
        "byte_count": candidate.stat().st_size,
        "production_pin": False,
    }
    assert start["details"]["vad_artifact"] == identity
    cohort = json.loads((root / "cohorts" / f"{_COHORT_ID}.private.json").read_bytes())
    assert cohort["vad_artifact"] == identity
    # The pinned sound tagger is part of the VAD the cohort replays.
    assert cohort["vad_assist"] == {
        "model_sha256": sound_tagger.MODEL_SHA256,
        "byte_count": sound_tagger.default_model_path().stat().st_size,
        "layout_sha256": canonical_sha256(vad_assist_layout()),
    }
    assert [Path(path) for path in sessions] == [candidate.resolve()]

    runs_before = sorted((root / "runs").iterdir())
    mismatched = write_request(
        "96000000-0000-4000-8000-000000000004",
        hashlib.sha256(b"a different silero graph").hexdigest(),
    )
    with pytest.raises(ValueError, match="vad_artifact"):
        run_request(mismatched, fixture.boundary, repo=_REPO, argv=("run",))
    assert sorted((root / "runs").iterdir()) == runs_before


def test_local_ct2_candidate_snapshot_is_recorded_file_by_file_in_run_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onnxruntime

    fixture = _verified_run(tmp_path, monkeypatch)
    root = fixture.boundary.root
    (master_path,) = (root / "masters").glob("*/master.private.json")
    master = json.loads(master_path.read_text("utf-8"))
    (reference_dir,) = (root / "references").iterdir()
    model_dir = tmp_path / "candidates" / "faster-whisper-ja-760M"
    model_dir.mkdir(parents=True)
    files = {"config.json": b"{}", "model.bin": b"candidate ct2 weights"}
    for name, payload in files.items():
        (model_dir / name).write_bytes(payload)
    config_path = root / "candidate.yaml"
    config_path.write_text(
        f"asr:\n  faster_whisper:\n    model: '{model_dir.as_posix()}'\n",
        "utf-8",
    )
    vad = tmp_path / "silero.onnx"
    vad.write_bytes(b"silero graph")
    request_path = root / "candidate.request.private.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": RUN_REQUEST_SCHEMA_VERSION,
                "request_id": "97000000-0000-4000-8000-000000000001",
                "master_id": master["master_id"],
                "reference_set_id": reference_dir.name,
                "config_path": str(config_path),
                "model_artifact_path": str(model_dir),
                "vad_artifact": {
                    "path": str(vad),
                    "sha256": hashlib.sha256(vad.read_bytes()).hexdigest(),
                },
                "arms": ["p0"],
                "arm_ids": {"p0": "97000000-0000-4000-8000-000000000002"},
                "cohort_id": "97000000-0000-4000-8000-000000000003",
                "cohort_fixture_ids": [master["fixture_id"]],
                "oracle_components": [],
            }
        ),
        "utf-8",
    )

    def stop_at_vad_load(_path: str, **_kwargs: object) -> None:
        raise RuntimeError("run-start is written before VAD or ASR construction")

    monkeypatch.setattr(onnxruntime, "InferenceSession", stop_at_vad_load)

    final = run_request(request_path, fixture.boundary, repo=_REPO, argv=("run",)).final

    start = json.loads((root / "runs" / final.run_id / "run-start.json").read_bytes())
    details = start["details"]
    assert details["model_artifact_path"] == str(model_dir.resolve())
    manifest = details["model_artifact_manifest"]
    assert manifest == {
        "kind": "tree",
        "files": [
            {
                "relative_path": name,
                "byte_sha256": hashlib.sha256(payload).hexdigest(),
                "byte_count": len(payload),
            }
            for name, payload in sorted(files.items())
        ],
    }
    assert canonical_sha256(manifest) == details["model_artifact_sha256"]


def test_dependency_version_that_differs_from_the_release_lock_is_refused_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_verify_release_lock", _REAL_VERIFY_RELEASE_LOCK)
    fixture = _verified_run(tmp_path, monkeypatch)
    root = fixture.boundary.root
    (master_path,) = (root / "masters").glob("*/master.private.json")
    master = json.loads(master_path.read_text("utf-8"))
    (reference_dir,) = (root / "references").iterdir()
    model_dir = tmp_path / "ct2-snapshot"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"synthetic ct2 weights")
    config_path = root / "config.yaml"
    config_path.write_text(
        f"asr:\n  faster_whisper:\n    model: '{model_dir.as_posix()}'\n",
        "utf-8",
    )
    vad = tmp_path / "silero.onnx"
    vad.write_bytes(b"silero graph")
    request_path = root / "lock.request.private.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": RUN_REQUEST_SCHEMA_VERSION,
                "request_id": "98000000-0000-4000-8000-000000000001",
                "master_id": master["master_id"],
                "reference_set_id": reference_dir.name,
                "config_path": str(config_path),
                "model_artifact_path": str(model_dir),
                "vad_artifact": {
                    "path": str(vad),
                    "sha256": hashlib.sha256(vad.read_bytes()).hexdigest(),
                },
                "arms": ["p0"],
                "arm_ids": {"p0": "98000000-0000-4000-8000-000000000002"},
                "cohort_id": "98000000-0000-4000-8000-000000000003",
                "cohort_fixture_ids": [master["fixture_id"]],
                "oracle_components": [],
            }
        ),
        "utf-8",
    )
    repo = tmp_path / "release-checkout"
    repo.mkdir()
    # Pin every checked dependency at its installed version except numpy, so
    # the refusal names numpy whatever the environment has installed.
    installed = runtime._dependency_identity()
    assert "numpy" in installed
    (repo / "constraints-release.txt").write_text(
        "".join(
            f"{name}=={'0.0.1' if name == 'numpy' else pinned}\n"
            for name, pinned in installed.items()
        ),
        "utf-8",
    )
    entries_before = sorted(root.rglob("*"))

    with pytest.raises(
        ValueError,
        match=r"^runtime dependency numpy==\S+ does not match the release lock 0\.0\.1$",
    ):
        run_request(request_path, fixture.boundary, repo=repo, argv=("run",))

    assert sorted(root.rglob("*")) == entries_before
