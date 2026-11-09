"""Shared pieces of the fine-tuning pilot (docs/ASR-FINETUNE.md).

Every step runs from the repository root with the pilot venv
(``<bench root>\\tools\\venv-ft``, built from ``requirements-ft.lock.txt``).
Bench decodes run in the release venv (``--bench-python``) through
``scripts.asr_bench.run``, the same path as any bench candidate, so a
converted model is judged exactly as production would run it; only
CTranslate2's sampler is seeded (``seeded_bench_run.py``), so a model decodes
the same texts every time.

Work directory layout (``--work-dir``, default ``<bench root>\\finetune-pilot``)::

    src\\whisper-ja-760M\\   transformers snapshot at SOURCE_REVISION
    ct2\\<label>\\           CTranslate2 conversions
    hf\\<label>\\            merged transformers checkpoints
    adapters\\<label>\\      LoRA adapters before the merge
    data\\cv-train5\\        the Common Voice clips of the training step
    candidates\\<label>.json bench candidates
    results\\<step>.json     each step's record
    reports\\<step>-<set_id>\\ bench scores (scripts.asr_bench.score)

Bench runs go to ``<runs dir>\\ft-pilot-<set_id>\\<label>`` (``--runs-dir``,
default ``<bench root>\\runs``); a step replaces the runs it writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sysconfig
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from scripts.asr_bench.sets import bench_root, load_set
from yamabiko.models import WHISPER_FILES
from yamabiko.paths import app_data_dir

REPO_ROOT = Path(__file__).resolve().parents[2]

# The transformers checkpoint behind the base export below. Its weights file
# is the same blob at every revision; this one carries the generation config
# whose suppress lists the base export's config.json repeats.
SOURCE_REPO = "efwkjn/whisper-ja-760M"
SOURCE_REVISION = "00daeaf3b79a85c5c52474147cc80b4370deae7f"
SOURCE_WEIGHTS = "model.safetensors"

# The base model's CTranslate2 export: its author's conversion of
# SOURCE_REPO, the shipped speech model until Kodama-1 (yamabiko.models)
# replaced it. The pilot rebuilds it (reproduce_export), checks every export
# against it (export) and decodes with it (teacher, cut_audit), so it is
# pinned here, not read from yamabiko.models. It holds the same five files
# (WHISPER_FILES).
BASE_EXPORT_REPO = "efwkjn/faster-whisper-ja-760M"
BASE_EXPORT_REVISION = "341201ba3e542985d2ad3f78dafb9c1665535013"
BASE_EXPORT_MODEL_BIN_SHA256 = "46ac4b1f0416a6637c84648d86fc0872d40a1f64fae87bfbb7eb1c176e4fabd2"

# ct2-transformers-converter settings of the base export.
CONVERTER_ARGS = (
    "--quantization", "float16",
    "--copy_files", "tokenizer.json", "preprocessor_config.json",
)

DEFAULT_SETS = ("probe-cv8", "silver-v5-tagonnx2")
RUN_STUDY = "ft-pilot"
BASE_LABEL = "base"
REPRO_LABEL = "repro"
# Seed of CTranslate2's sampler in every pilot decode (seeded_bench_run.py).
CT2_SEED = 1

# include/ctranslate2/types.h DataType order (ctranslate2.specs.model_spec).
_CT2_DTYPES = ("float32", "int8", "int16", "int32", "float16", "bfloat16")
_CT2_BINARY_VERSION = 6
_HASH_CHUNK = 1 << 24


# Paths and arguments -----------------------------------------------------------


def default_work_dir() -> Path:
    return bench_root() / "finetune-pilot"


def add_common_args(parser: argparse.ArgumentParser, *, bench: bool = True) -> None:
    parser.add_argument("--work-dir", type=Path, default=default_work_dir(),
                        help="pilot work directory (default: <bench root>\\finetune-pilot)")
    if not bench:
        return
    parser.add_argument("--bench-python", type=Path, default=_default_bench_python(),
                        help="release venv interpreter for scripts.asr_bench.run "
                             "(default: <repo>\\.venv\\Scripts\\python.exe when it exists)")
    parser.add_argument("--set", dest="sets", type=Path, action="append",
                        help=f"bench set directory, repeatable (default: {', '.join(DEFAULT_SETS)} "
                             "under <bench root>\\sets)")
    parser.add_argument("--runs-dir", type=Path, default=bench_root() / "runs",
                        help="bench runs root (default: <bench root>\\runs)")


def _default_bench_python() -> Path | None:
    candidate = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    return candidate if candidate.is_file() else None


def resolve_bench_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[Path]:
    """Check ``--bench-python`` and return the set directories."""
    if args.bench_python is None or not Path(args.bench_python).is_file():
        parser.error("--bench-python must name the release venv interpreter")
    sets = args.sets or [bench_root() / "sets" / name for name in DEFAULT_SETS]
    for set_dir in sets:
        if not (Path(set_dir) / "set.json").is_file():
            parser.error(f"no bench set at {set_dir}")
    return [Path(set_dir) for set_dir in sets]


def source_dir(work_dir: Path) -> Path:
    return work_dir / "src" / "whisper-ja-760M"


def ct2_dir(work_dir: Path, label: str) -> Path:
    return work_dir / "ct2" / label


def hf_dir(work_dir: Path, label: str) -> Path:
    return work_dir / "hf" / label


def require_source(work_dir: Path) -> Path:
    source = source_dir(work_dir)
    if not (source / SOURCE_WEIGHTS).is_file():
        raise SystemExit(f"{source} holds no {SOURCE_WEIGHTS}: run scripts.asr_finetune.reproduce_export first")
    return source


def download_source(work_dir: Path) -> Path:
    """The transformers snapshot at SOURCE_REVISION (downloaded once)."""
    return fetch_source(source_dir(work_dir))


def fetch_source(target: Path) -> Path:
    """The transformers snapshot at SOURCE_REVISION in ``target`` (current
    files already there are kept)."""
    from huggingface_hub import snapshot_download

    snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION, local_dir=target)
    return target


def base_export_dir(*, download: bool = False) -> Path:
    """The base export (``BASE_EXPORT_REPO``) in the app's model cache, at
    ``BASE_EXPORT_REVISION`` (fetched into that cache first with
    ``download``, as on a pod)."""
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(
        BASE_EXPORT_REPO, revision=BASE_EXPORT_REVISION, allow_patterns=list(WHISPER_FILES),
        cache_dir=app_data_dir() / "hf-cache" / "hub", local_files_only=not download,
    ))
    problem = not_base_export(path)
    if problem:
        raise SystemExit(problem)
    return path


def not_base_export(path: Path) -> str | None:
    """Why ``path`` is not the pinned base export, or None when its
    ``model.bin`` is the pinned one (``BASE_EXPORT_MODEL_BIN_SHA256``)."""
    digest = sha256_file(path / "model.bin")
    if digest != BASE_EXPORT_MODEL_BIN_SHA256:
        return f"{path / 'model.bin'}: SHA-256 {digest} is not the pinned {BASE_EXPORT_MODEL_BIN_SHA256}"
    return None


# Hashing and results -----------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def write_result(work_dir: Path, step: str, record: Mapping[str, Any]) -> Path:
    path = work_dir / "results" / f"{step}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"step": step, "created_at_utc": datetime.now(timezone.utc).isoformat(), **record}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def package_versions() -> dict[str, str]:
    from importlib.metadata import version

    names = ("torch", "transformers", "peft", "ctranslate2", "safetensors", "huggingface-hub", "numpy")
    return {name: version(name) for name in names}


# CTranslate2 model.bin ---------------------------------------------------------


@dataclass(frozen=True)
class Ct2Variable:
    shape: tuple[int, ...]
    dtype: str
    offset: int
    nbytes: int


@dataclass(frozen=True)
class Ct2Model:
    binary_version: int
    spec: str
    spec_revision: int
    variables: dict[str, Ct2Variable]
    aliases: dict[str, str]


def _unpack(handle: BinaryIO, fmt: str) -> int:
    size = struct.calcsize(fmt)
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("truncated model.bin")
    return struct.unpack(fmt, data)[0]


def _string(handle: BinaryIO) -> str:
    length = _unpack(handle, "<H")
    data = handle.read(length)
    if len(data) != length or data[-1:] != b"\0":
        raise ValueError("malformed string in model.bin")
    return data[:-1].decode("utf-8")


def read_ct2_model(path: Path) -> Ct2Model:
    """Index a CTranslate2 ``model.bin`` (binary version 6) without loading
    its payload: the layout ``ModelSpec._serialize`` writes."""
    with path.open("rb") as handle:
        version = _unpack(handle, "<I")
        if version != _CT2_BINARY_VERSION:
            raise ValueError(f"{path}: binary version {version}, expected {_CT2_BINARY_VERSION}")
        spec = _string(handle)
        revision = _unpack(handle, "<I")
        variables: dict[str, Ct2Variable] = {}
        for _ in range(_unpack(handle, "<I")):
            name = _string(handle)
            shape = tuple(_unpack(handle, "<I") for _ in range(_unpack(handle, "<B")))
            dtype = _CT2_DTYPES[_unpack(handle, "<B")]
            nbytes = _unpack(handle, "<I")
            variables[name] = Ct2Variable(shape, dtype, handle.tell(), nbytes)
            handle.seek(nbytes, 1)
        aliases: dict[str, str] = {}
        for _ in range(_unpack(handle, "<I")):
            alias = _string(handle)
            aliases[alias] = _string(handle)
        if handle.read(1):
            raise ValueError(f"{path}: trailing bytes after the alias table")
    return Ct2Model(version, spec, revision, variables, aliases)


def _as_values(raw: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bfloat16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return raw.view(np.dtype(dtype))


def compare_values(reference: np.ndarray, other: np.ndarray) -> dict[str, Any]:
    """Element statistics of two same-shape float arrays whose bytes differ."""
    a = reference.astype(np.float64)
    b = other.astype(np.float64)
    differ = a != b
    bit_differ = int(np.count_nonzero(reference.view(f"u{reference.itemsize}") != other.view(f"u{other.itemsize}")))
    delta = np.abs(a - b)[differ]
    scale = float(np.max(np.abs(a))) if a.size else 0.0
    return {
        "elements": int(a.size),
        "elements_bits_differ": bit_differ,
        "elements_value_differ": int(np.count_nonzero(differ)),
        "signed_zero_only": bit_differ - int(np.count_nonzero(differ)),
        "max_abs_diff": float(delta.max()) if delta.size else 0.0,
        "max_abs_reference": scale,
    }


def value_identical(differing: Mapping[str, Mapping[str, Any]]) -> bool:
    """True when every byte difference is a zero whose sign flipped."""
    return all(entry.get("elements_value_differ", 1) == 0 for entry in differing.values())


def weight_summary(record: Mapping[str, Any]) -> str:
    """One line for a ``compare_ct2`` or ``lora.compare_safetensors`` record."""
    total = record.get("variables", record.get("tensors"))
    identical = record.get("identical_variables", record.get("identical_tensors"))
    differing = record.get("differing_variables", record.get("differing_tensors", {}))
    changed = sum(entry.get("elements_value_differ", 0) for entry in differing.values())
    flips = sum(entry.get("signed_zero_only", 0) for entry in differing.values())
    largest = max((entry.get("max_abs_diff", 0.0) for entry in differing.values()), default=0.0)
    return (f"{identical}/{total} byte-identical; {len(differing)} differ: {changed:,} elements by value "
            f"(max |diff| {largest:.3g}), {flips:,} only by the sign of a zero")


def compare_ct2(reference: Path, other: Path) -> dict[str, Any]:
    """Variable-by-variable comparison of two ``model.bin`` files."""
    a = read_ct2_model(reference)
    b = read_ct2_model(other)
    map_a = np.memmap(reference, dtype=np.uint8, mode="r")
    map_b = np.memmap(other, dtype=np.uint8, mode="r")
    common = sorted(set(a.variables) & set(b.variables))
    identical = 0
    differing: dict[str, Any] = {}
    for name in common:
        va, vb = a.variables[name], b.variables[name]
        if (va.shape, va.dtype) != (vb.shape, vb.dtype):
            differing[name] = {"reference": [list(va.shape), va.dtype], "other": [list(vb.shape), vb.dtype]}
            continue
        raw_a = map_a[va.offset:va.offset + va.nbytes]
        raw_b = map_b[vb.offset:vb.offset + vb.nbytes]
        if np.array_equal(raw_a, raw_b):
            identical += 1
            continue
        if va.dtype in ("float16", "bfloat16", "float32"):
            differing[name] = compare_values(_as_values(raw_a, va.dtype), _as_values(raw_b, vb.dtype))
        else:
            differing[name] = {"bytes_differ": int(np.count_nonzero(raw_a != raw_b))}
    dtypes: dict[str, int] = {}
    for variable in a.variables.values():
        dtypes[variable.dtype] = dtypes.get(variable.dtype, 0) + 1
    return {
        "header_identical": (a.binary_version, a.spec, a.spec_revision) == (b.binary_version, b.spec, b.spec_revision),
        "spec": f"{a.spec} revision {a.spec_revision}, binary version {a.binary_version}",
        "variables": len(a.variables),
        "reference_dtypes": dtypes,
        "only_in_reference": sorted(set(a.variables) - set(b.variables)),
        "only_in_other": sorted(set(b.variables) - set(a.variables)),
        "order_identical": list(a.variables) == list(b.variables),
        "identical_variables": identical,
        "value_identical": value_identical(differing),
        "differing_variables": differing,
        "aliases_identical": a.aliases == b.aliases,
    }


def compare_model_dirs(reference: Path, other: Path) -> dict[str, Any]:
    """File-by-file SHA-256 of two CTranslate2 directories, plus the
    ``model.bin`` variable comparison when the weights differ."""
    names = sorted(
        {path.name for path in reference.iterdir() if path.is_file()}
        | {path.name for path in other.iterdir() if path.is_file()}
    )
    files: dict[str, Any] = {}
    for name in names:
        a, b = reference / name, other / name
        entry = {
            "reference_sha256": sha256_file(a) if a.is_file() else None,
            "sha256": sha256_file(b) if b.is_file() else None,
            "size": b.stat().st_size if b.is_file() else None,
        }
        entry["identical"] = entry["reference_sha256"] is not None and entry["reference_sha256"] == entry["sha256"]
        if name.endswith(".json") and not entry["identical"] and a.is_file() and b.is_file():
            entry["json_equal"] = json.loads(a.read_text(encoding="utf-8")) == json.loads(b.read_text(encoding="utf-8"))
        files[name] = entry
    record: dict[str, Any] = {
        "reference": str(reference),
        "other": str(other),
        "files": files,
        "all_identical": all(entry["identical"] for entry in files.values()),
    }
    if (reference / "model.bin").is_file() and (other / "model.bin").is_file():
        record["model_bin"] = compare_ct2(reference / "model.bin", other / "model.bin")
    return record


def _canonical_json(data: bytes) -> str | None:
    """The JSON content as one canonical string (keys sorted, no
    whitespace, ``true`` apart from ``1``), or None when it is no JSON."""
    try:
        return json.dumps(json.loads(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except ValueError:
        return None


def adopt_reference_json(reference: Path, out: Path) -> dict[str, dict[str, Any]]:
    """Give each ``.json`` file of the CTranslate2 directory ``out`` whose
    content equals the reference's but whose bytes do not the reference
    file's bytes. Some converter builds re-serialize ``config.json`` and
    ``vocabulary.json`` (other whitespace and escaping, the same JSON), so
    the directory then matches the reference byte for byte outside
    ``model.bin``. A file whose content differs stays as converted. Returns
    the replaced files with the converter's SHA-256 and size."""
    replaced: dict[str, dict[str, Any]] = {}
    for path in sorted(out.glob("*.json")):
        source = reference / path.name
        if not source.is_file():
            continue
        converted, shipped = path.read_bytes(), source.read_bytes()
        if converted == shipped:
            continue
        content = _canonical_json(converted)
        if content is None or content != _canonical_json(shipped):
            continue
        shutil.copyfile(source, path)
        replaced[path.name] = {"converted_sha256": hashlib.sha256(converted).hexdigest(),
                               "converted_size": len(converted)}
    return replaced


def ct2_files_equal(files: Mapping[str, Any]) -> bool:
    """Every file of a ``compare_model_dirs`` record but ``model.bin`` equal
    to the reference: byte-identical, or ``config.json`` the same JSON."""
    return all(entry["identical"] or (name == "config.json" and entry.get("json_equal", False))
               for name, entry in files["files"].items() if name != "model.bin")


# Conversion --------------------------------------------------------------------


def convert_to_ct2(source: Path, out: Path) -> list[str]:
    """Convert a transformers Whisper directory with the base export's
    settings (``CONVERTER_ARGS``)."""
    converter = shutil.which("ct2-transformers-converter", path=sysconfig.get_path("scripts"))
    if converter is None:
        raise SystemExit("ct2-transformers-converter is not installed in this venv")
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [converter, "--model", str(source), "--output_dir", str(out), *CONVERTER_ARGS]
    subprocess.run(command, check=True)
    return ["ct2-transformers-converter", "--model", "<src>", "--output_dir", "<out>", *CONVERTER_ARGS]


# Bench decodes -----------------------------------------------------------------


def bench_candidate(label: str, model: str) -> dict[str, Any]:
    """The bench's 760M baseline (``c760-base``): shipped chunker and ASR
    settings, float16, beam 5."""
    return {
        "name": label,
        "vad": None,
        "asr": {"backend": "faster_whisper", "model": model, "compute_type": "float16", "beam_size": 5},
    }


def run_dir(runs_dir: Path, set_id: str, label: str) -> Path:
    return runs_dir / f"{RUN_STUDY}-{set_id}" / label


def bench_decode(
    *, label: str, model: str, sets: Sequence[Path], runs_dir: Path, bench_python: Path, work_dir: Path,
) -> dict[str, Path]:
    """Run ``label`` over every set with ``scripts.asr_bench.run``, the
    sampler seeded with ``CT2_SEED``."""
    candidate_path = work_dir / "candidates" / f"{label}.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps(bench_candidate(label, model), indent=1) + "\n", encoding="utf-8")
    runs: dict[str, Path] = {}
    for set_dir in sets:
        set_id = load_set(set_dir).set_id
        out = run_dir(runs_dir, set_id, label)
        if out.exists():
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(bench_python), "-m", "scripts.asr_finetune.seeded_bench_run", "--seed", str(CT2_SEED),
             "--set", str(set_dir), "--candidate", str(candidate_path), "--out", str(out)],
            cwd=REPO_ROOT, check=True,
        )
        runs[set_id] = out
    return runs


def existing_runs(runs_dir: Path, sets: Sequence[Path], label: str) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for set_dir in sets:
        set_id = load_set(set_dir).set_id
        path = run_dir(runs_dir, set_id, label)
        if not (path / "windows.jsonl").is_file():
            raise SystemExit(f"no {label} run at {path}: run scripts.asr_finetune.reproduce_export first")
        runs[set_id] = path
    return runs


def _windows(path: Path) -> list[dict[str, Any]]:
    with (path / "windows.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_model_identity(path: Path) -> dict[str, Any]:
    record = json.loads((path / "run.json").read_text(encoding="utf-8"))
    return record["asr"]["model"]


def compare_runs(reference: Path, other: Path) -> dict[str, Any]:
    """Window-by-window texts of two runs over the same set."""
    a = _windows(reference)
    b = _windows(other)
    span = ("item_id", "window_index", "start_s", "end_s", "emit_reason")
    spans_identical = len(a) == len(b) and all(
        tuple(x[key] for key in span) == tuple(y[key] for key in span) for x, y in zip(a, b)
    )
    differing = [
        {"item_id": x["item_id"], "window_index": x["window_index"], "reference": x["text"], "text": y["text"]}
        for x, y in zip(a, b)
        if x["text"] != y["text"] or x["asr_outcome"] != y["asr_outcome"]
    ]
    return {
        "reference": str(reference),
        "other": str(other),
        "reference_model": run_model_identity(reference),
        "other_model": run_model_identity(other),
        "windows": len(a),
        "spans_identical": spans_identical,
        "identical_texts": min(len(a), len(b)) - len(differing),
        "differing": differing,
    }


def compare_run_sets(reference: Mapping[str, Path], other: Mapping[str, Path]) -> dict[str, Any]:
    return {set_id: compare_runs(reference[set_id], other[set_id]) for set_id in reference}


def text_summary(comparisons: Mapping[str, Mapping[str, Any]]) -> str:
    return ", ".join(
        f"{set_id} {record['identical_texts']}/{record['windows']}"
        + ("" if record["spans_identical"] else " (spans differ)")
        for set_id, record in comparisons.items()
    )


_SCORE_KEYS = ("aligned_recall_reading", "cer", "cer_reading", "keyword_recall", "windows")


def bench_score(
    runs: Sequence[Path], *, out: Path, bench_python: Path,
) -> dict[str, dict[str, Any]]:
    """Score runs over one set with ``scripts.asr_bench.score`` (the first
    is the baseline) and return each candidate's headline figures."""
    if out.exists():
        shutil.rmtree(out)
    command = [str(bench_python), "-m", "scripts.asr_bench.score"]
    for path in runs:
        command += ["--run", str(path)]
    # The scorer prints report.md, which it also writes into ``out``.
    subprocess.run([*command, "--out", str(out)], cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    (scored,) = report["sets"]
    return {
        label: {key: groups["all"][key] for key in _SCORE_KEYS if key in groups["all"]}
        for label, groups in scored["metrics"].items()
    }
