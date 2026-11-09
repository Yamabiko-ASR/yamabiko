"""Privacy-safe command line surface for the E0 evidence harness."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NoReturn

from ._common import require_uuid4
from .ledger import IncompleteRunError, verify_run
from .private_root import PrivateBoundary
from .schema import ArmKind, Partition, RunStatus


_PROGRAM = "python -m scripts.e0_evidence"


class _CliInputError(ValueError):
    """An invalid command input whose original value must not be rendered."""


class _SafeArgumentParser(argparse.ArgumentParser):
    """Argument parser that never echoes a rejected private argument."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> NoReturn:
        del message
        raise _CliInputError("invalid command input")


Handler = Callable[[argparse.Namespace], int]


def _absolute_path(raw: object) -> Path:
    if type(raw) is not str or not raw:
        raise _CliInputError("invalid absolute path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise _CliInputError("invalid absolute path")
    try:
        return candidate.resolve(strict=False)
    except OSError as exc:
        raise _CliInputError("invalid absolute path") from exc


def _opaque_id(raw: object) -> str:
    try:
        return require_uuid4(raw, "opaque ID")
    except (TypeError, ValueError) as exc:
        raise _CliInputError("invalid opaque ID") from exc


def _excluded_roots(args: argparse.Namespace) -> tuple[Path, ...]:
    return tuple(_absolute_path(value) for value in args.exclude_root)


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _boundary(args: argparse.Namespace) -> PrivateBoundary:
    from .registry import boundary_from_environment

    return boundary_from_environment(
        _repo(),
        os.environ,
        excluded_roots=_excluded_roots(args),
    )


def _emit(fields: Sequence[tuple[str, object]]) -> None:
    rendered = " ".join(f"{name}={value}" for name, value in fields)
    sys.stdout.write(f"{rendered}\n")


def _emit_error(exc: BaseException, *, status: str = "invalid") -> None:
    _emit_to_stderr(
        (
            ("status", status),
            ("error", type(exc).__name__.lstrip("_")),
        )
    )


def _emit_to_stderr(fields: Sequence[tuple[str, object]]) -> None:
    rendered = " ".join(f"{name}={value}" for name, value in fields)
    sys.stderr.write(f"{rendered}\n")


def _handle_root_init(args: argparse.Namespace) -> int:
    from .registry import initialize_private_root

    initialize_private_root(
        _repo(),
        os.environ,
        excluded_roots=_excluded_roots(args),
    )
    _emit((("status", "initialized"),))
    return 0


def _handle_fixture_register(args: argparse.Namespace) -> int:
    from .assets import register_fixture

    title_id = None if args.title_id is None else _opaque_id(args.title_id)
    manifest = register_fixture(
        _absolute_path(args.source),
        Partition(args.partition),
        _boundary(args),
        title_id=title_id,
    )
    _emit(
        (
            ("fixture_id", manifest.fixture_id),
            ("title_id", manifest.title_id),
            ("status", "registered"),
        )
    )
    return 0


def _handle_rights_apply(args: argparse.Namespace) -> int:
    from .registry import apply_rights_review

    rights = apply_rights_review(
        _absolute_path(args.review),
        _boundary(args),
    )
    _emit(
        (
            ("fixture_id", rights.fixture_id),
            ("rights_record_id", rights.rights_record_id),
            ("status", "applied"),
        )
    )
    return 0


def _handle_master_register(args: argparse.Namespace) -> int:
    from .registry import register_master

    manifest = register_master(
        _absolute_path(args.manifest),
        _absolute_path(args.source_native),
        _absolute_path(args.canonical),
        _boundary(args),
    )
    _emit(
        (
            ("fixture_id", manifest.fixture_id),
            ("master_id", manifest.master_id),
            ("status", "registered"),
        )
    )
    return 0


def _handle_annotation_pass_import(args: argparse.Namespace) -> int:
    from .annotations import import_annotation_pass, load_registered_annotation_pass

    boundary = _boundary(args)
    editorial_sidecar = (
        None
        if args.editorial_sidecar is None
        else _absolute_path(args.editorial_sidecar)
    )
    bundle = import_annotation_pass(
        _absolute_path(args.worksheet),
        _absolute_path(args.work),
        args.reference_version,
        _opaque_id(args.master_id),
        boundary,
        editorial_sidecar_path=editorial_sidecar,
    )
    registered = load_registered_annotation_pass(bundle.pass_id, boundary)
    _emit((("pass_id", bundle.pass_id), ("status", registered.status)))
    return 0


def _handle_annotation_proposal_create(args: argparse.Namespace) -> int:
    from .subtitle_proposals import create_subtitle_proposal

    artifacts = create_subtitle_proposal(
        _opaque_id(args.master_id),
        args.subtitle_stream,
        _absolute_path(args.ffmpeg),
        _absolute_path(args.worksheet),
        _absolute_path(args.editorial_sidecar),
        _boundary(args),
    )
    _emit((("status", "created"),))
    _emit((("cues", artifacts.cue_count),))
    return 0


def _handle_reference_draft_from_pass(args: argparse.Namespace) -> int:
    from .annotations import assemble_reference_draft_from_pass

    manifest = assemble_reference_draft_from_pass(
        _opaque_id(args.pass_id),
        _opaque_id(args.reference_set_id),
        _absolute_path(args.draft),
        _absolute_path(args.text),
        _boundary(args),
    )
    _emit(
        (
            ("fixture_id", manifest.fixture_id),
            ("reference_set_id", manifest.reference_set_id),
            ("status", "draft"),
        )
    )
    return 0


def _reference_arguments(
    args: argparse.Namespace,
) -> tuple[Path, Path, str, PrivateBoundary]:
    return (
        _absolute_path(args.draft),
        _absolute_path(args.text),
        _opaque_id(args.master_id),
        _boundary(args),
    )


def _handle_reference_validate(args: argparse.Namespace) -> int:
    from .registry import validate_reference_draft

    manifest = validate_reference_draft(*_reference_arguments(args))
    _emit(
        (
            ("fixture_id", manifest.fixture_id),
            ("reference_set_id", manifest.reference_set_id),
            ("status", "valid"),
        )
    )
    return 0


def _handle_reference_freeze(args: argparse.Namespace) -> int:
    from .registry import freeze_reference_draft

    manifest = freeze_reference_draft(*_reference_arguments(args))
    _emit(
        (
            ("fixture_id", manifest.fixture_id),
            ("reference_set_id", manifest.reference_set_id),
            ("status", "frozen"),
        )
    )
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    from .runtime import run_request

    outcome = run_request(
        _absolute_path(args.request),
        _boundary(args),
        repo=_repo(),
        argv=args.command_argv,
    )
    for failure in outcome.arm_failures:
        _emit_to_stderr(
            (
                ("arm_kind", failure.arm_kind.value),
                ("status", "failed"),
                ("reason", failure.reason),
                ("exception", failure.exception_class),
            )
        )
    final = outcome.final
    complete = final.status is RunStatus.COMPLETE
    _emit(
        (
            ("run_id", final.run_id),
            ("status", "complete" if complete else "incomplete"),
            ("reasons", len(final.incomplete_reasons)),
        )
    )
    return 0 if complete else 1


def _handle_verify_run(args: argparse.Namespace) -> int:
    run_dir = _absolute_path(args.run_dir)
    if not run_dir.is_dir():
        raise _CliInputError("run directory is unavailable")
    report = verify_run(run_dir)
    _emit(
        (
            ("run_id", report.run_id or "unavailable"),
            ("status", "complete" if report.complete else "incomplete"),
            ("reasons", len(report.reasons)),
        )
    )
    return 0 if report.complete else 1


def _handle_score(args: argparse.Namespace) -> int:
    from .score_runs import write_run_scores

    boundary = _boundary(args)
    run_ids = tuple(_opaque_id(value) for value in args.run_id)
    bundle = write_run_scores(
        tuple(boundary.root / "runs" / run_id for run_id in run_ids),
        boundary,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    fields: list[tuple[str, object]] = [
        ("report_id", bundle.report_id),
        ("status", "complete"),
    ]
    if bundle.omitted_stochastic_arm_ids:
        fields.append(("p2b", "omitted_stochastic"))
    _emit(fields)
    return 0


def _handle_report(args: argparse.Namespace) -> int:
    from .report import write_private_report_from_bundle

    artifacts = write_private_report_from_bundle(
        _absolute_path(args.report_dir),
        _boundary(args),
        write_views=args.write_views,
    )
    _emit(
        (
            ("status", "complete"),
            ("views", len(artifacts.derived_views)),
        )
    )
    return 0


def _handle_compare(args: argparse.Namespace) -> int:
    from .compare import CONTRAST_METRICS, compare_reports

    candidate_ids = tuple(_opaque_id(value) for value in args.candidate_report)
    comparison_dir = compare_reports(
        _opaque_id(args.baseline_report),
        candidate_ids,
        _boundary(args),
        arm_kind=ArmKind(args.arm_kind),
        replicates=args.bootstrap_replicates,
    )
    _emit(
        (
            ("comparison_id", comparison_dir.name),
            ("status", "complete"),
            ("contrasts", len(CONTRAST_METRICS) * len(candidate_ids)),
        )
    )
    return 0


def _handle_export_public(args: argparse.Namespace) -> int:
    from .report import export_public

    artifacts = export_public(
        _absolute_path(args.report_dir),
        _absolute_path(args.review),
        _absolute_path(args.output),
        _boundary(args),
    )
    _emit(
        (
            ("receipt_id", artifacts.receipt_id),
            ("status", "complete"),
        )
    )
    return 0


def _add_boundary_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--exclude-root",
        action="append",
        default=[],
        metavar="ABS_PATH",
        help="absolute root that private operations must not enter",
    )


def _command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    handler: Handler,
    *,
    boundary: bool = True,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name)
    if boundary:
        _add_boundary_option(parser)
    parser.set_defaults(handler=handler)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(prog=_PROGRAM)
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_SafeArgumentParser,
    )

    _command(subparsers, "root-init", _handle_root_init)

    fixture = _command(
        subparsers,
        "fixture-register",
        _handle_fixture_register,
    )
    fixture.add_argument("--source", required=True, metavar="ABS_PATH")
    fixture.add_argument(
        "--partition",
        required=True,
        choices=tuple(item.value for item in Partition),
    )
    fixture.add_argument("--title-id", metavar="UUID4")

    rights = _command(subparsers, "rights-apply", _handle_rights_apply)
    rights.add_argument("--review", required=True, metavar="ABS_PATH")

    master = _command(
        subparsers,
        "master-register",
        _handle_master_register,
    )
    master.add_argument("--manifest", required=True, metavar="ABS_PATH")
    master.add_argument("--source-native", required=True, metavar="ABS_PATH")
    master.add_argument("--canonical", required=True, metavar="ABS_PATH")

    annotation_pass = _command(
        subparsers,
        "annotation-pass-import",
        _handle_annotation_pass_import,
    )
    annotation_pass.add_argument("--worksheet", required=True, metavar="ABS_PATH")
    annotation_pass.add_argument("--work", required=True, metavar="ABS_PATH")
    annotation_pass.add_argument("--reference-version", required=True)
    annotation_pass.add_argument("--master-id", required=True, metavar="UUID4")
    annotation_pass.add_argument("--editorial-sidecar", metavar="ABS_PATH")

    annotation_proposal = _command(
        subparsers,
        "annotation-proposal-create",
        _handle_annotation_proposal_create,
    )
    annotation_proposal.add_argument("--master-id", required=True, metavar="UUID4")
    annotation_proposal.add_argument(
        "--subtitle-stream",
        required=True,
        type=int,
        metavar="N",
    )
    annotation_proposal.add_argument("--ffmpeg", required=True, metavar="ABS_PATH")
    annotation_proposal.add_argument("--worksheet", required=True, metavar="ABS_PATH")
    annotation_proposal.add_argument(
        "--editorial-sidecar",
        required=True,
        metavar="ABS_PATH",
    )

    reference_draft = _command(
        subparsers,
        "reference-draft-from-pass",
        _handle_reference_draft_from_pass,
    )
    reference_draft.add_argument("--pass-id", required=True, metavar="UUID4")
    reference_draft.add_argument(
        "--reference-set-id",
        required=True,
        metavar="UUID4",
    )
    reference_draft.add_argument("--draft", required=True, metavar="ABS_PATH")
    reference_draft.add_argument("--text", required=True, metavar="ABS_PATH")

    for name, handler in (
        ("reference-validate", _handle_reference_validate),
        ("reference-freeze", _handle_reference_freeze),
    ):
        reference = _command(subparsers, name, handler)
        reference.add_argument("--draft", required=True, metavar="ABS_PATH")
        reference.add_argument("--text", required=True, metavar="ABS_PATH")
        reference.add_argument("--master-id", required=True, metavar="UUID4")

    run = _command(subparsers, "run", _handle_run)
    run.add_argument("--request", required=True, metavar="ABS_PATH")

    verify = _command(
        subparsers,
        "verify-run",
        _handle_verify_run,
        boundary=False,
    )
    verify.add_argument("--run-dir", required=True, metavar="ABS_PATH")

    score = _command(subparsers, "score", _handle_score)
    score.add_argument("--run-id", action="append", required=True, metavar="UUID4")
    score.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
        metavar="N",
    )

    report = _command(subparsers, "report", _handle_report)
    report.add_argument("--report-dir", required=True, metavar="ABS_PATH")
    report.add_argument("--write-views", action="store_true")

    compare = _command(subparsers, "compare", _handle_compare)
    compare.add_argument("--baseline-report", required=True, metavar="UUID4")
    compare.add_argument(
        "--candidate-report",
        action="append",
        required=True,
        metavar="UUID4",
    )
    compare.add_argument(
        "--arm-kind",
        choices=(ArmKind.P0.value, ArmKind.P1.value),
        default=ArmKind.P0.value,
    )
    compare.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
        metavar="N",
    )

    export = _command(
        subparsers,
        "export-public",
        _handle_export_public,
    )
    export.add_argument("--report-dir", required=True, metavar="ABS_PATH")
    export.add_argument("--review", required=True, metavar="ABS_PATH")
    export.add_argument("--output", required=True, metavar="ABS_PATH")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    command_argv = tuple(
        getattr(sys, "orig_argv", (sys.executable, *sys.argv))
        if argv is None
        else (
            sys.executable,
            "-m",
            "scripts.e0_evidence",
            *raw_argv,
        )
    )
    try:
        args = build_parser().parse_args(raw_argv)
        args.command_argv = command_argv
        return args.handler(args)
    except SystemExit as exc:
        return int(exc.code or 0)
    except IncompleteRunError as exc:
        _emit_error(exc, status="incomplete")
        return 1
    except KeyboardInterrupt as exc:
        _emit_error(exc, status="cancelled")
        return 1
    except (ValueError, TypeError, OSError, UnicodeError) as exc:
        _emit_error(exc)
        return 2
    except Exception as exc:
        _emit_error(exc, status="failed")
        return 1


__all__ = ["build_parser", "main"]
