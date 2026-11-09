"""Unit tests for yamabiko.lifecycle.exit_codes."""
from __future__ import annotations

import pytest

from yamabiko.lifecycle.exit_codes import (
    EXIT_CLEAN,
    EXIT_WIZARD_CANCEL,
    EXIT_TAILORED_ERROR,
    EXIT_STREAM_DEATH,
    EXIT_STAGE_ESCALATION,
    EXIT_STARTUP_FAILURE,
    EXIT_CHILD_GIVEUP,
    ExitPolicy,
    classify,
)


@pytest.mark.parametrize("code, expected", [
    (EXIT_CLEAN, ExitPolicy.STOP_SELF),
    (EXIT_WIZARD_CANCEL, ExitPolicy.STOP_ALL),
    (EXIT_TAILORED_ERROR, ExitPolicy.STOP_ALL),
    (EXIT_STARTUP_FAILURE, ExitPolicy.STOP_ALL),
    (EXIT_CHILD_GIVEUP, ExitPolicy.RESTART),
    (1, ExitPolicy.RESTART),
    (2, ExitPolicy.RESTART),
    (139, ExitPolicy.RESTART),   # SIGSEGV on POSIX; some Windows crashes
    (-1073741819, ExitPolicy.RESTART),  # 0xC0000005 access violation
])
def test_classify_table(code: int, expected: ExitPolicy):
    """Fails red if: exit code is misclassified (esp. 43 regressing
    to RESTART)."""
    assert classify(code) is expected


def test_stream_death_code_classifies_as_restart():
    """Fails red if: EXIT_STREAM_DEATH (the capture-stream-death restart
    signal) is moved into the STOP_ALL tuple. A dead / unplugged / silent
    capture device must make the watchdog RESTART and re-resolve the device,
    NOT stop the session. Pins the value too -- it is an inter-process
    contract (the pipeline child raises it, the watchdog parent classifies
    it), so the integer itself is load-bearing.

    Seeded defect: add EXIT_STREAM_DEATH to the (EXIT_WIZARD_CANCEL,
    EXIT_TAILORED_ERROR) tuple in classify() -> STOP_ALL -> red.
    """
    assert EXIT_STREAM_DEATH == 44
    assert classify(EXIT_STREAM_DEATH) is ExitPolicy.RESTART


def test_stage_escalation_code_classifies_as_restart():
    """Fails red if: EXIT_STAGE_ESCALATION (a pipeline stage crossed its
    supervisor escalation threshold, e.g. translation failing 10x in 60 s)
    is moved into the STOP_ALL tuple -- or removed so escalation regresses
    to exit 0. classify(0) is STOP_SELF: the watchdog would read a stage
    crash-loop as a clean quit and cascade a full stop with no dialog and
    no restart (the silent-death bug this code exists to fix). Pins the
    value too -- it is an inter-process contract (the pipeline child
    raises it, the watchdog parent classifies it), so the integer itself
    is load-bearing.

    Seeded defect: add EXIT_STAGE_ESCALATION to the (EXIT_WIZARD_CANCEL,
    EXIT_TAILORED_ERROR) tuple in classify() -> STOP_ALL -> red.
    """
    assert EXIT_STAGE_ESCALATION == 45
    assert classify(EXIT_STAGE_ESCALATION) is ExitPolicy.RESTART


def test_startup_failure_code_classifies_as_stop_all():
    """Fails red if: EXIT_STARTUP_FAILURE (the overlay-server port-bind
    failure) is removed from the STOP_ALL tuple -- or its value drifts.
    A bind conflict does not self-heal in seconds, so the watchdog must
    give up (STOP_ALL: cascade peers + mutex drop, no restart churn) and
    show the port dialog, NOT route it to RESTART (a crash-loop that
    just delays the message) nor STOP_SELF (a silent clean-quit cascade,
    the exact silent-death bug this code exists to fix). Pins the value:
    it is an inter-process contract (pipeline child raises it, watchdog
    parent classifies it).

    Seeded defect: drop EXIT_STARTUP_FAILURE from the STOP_ALL tuple in
    classify() -> classify returns RESTART -> red.
    """
    assert EXIT_STARTUP_FAILURE == 46
    assert classify(EXIT_STARTUP_FAILURE) is ExitPolicy.STOP_ALL


def test_child_giveup_code_classifies_as_restart():
    """Fails red if: EXIT_CHILD_GIVEUP (the overlay supervisor exhausted
    its respawn budget) is moved INTO the STOP_ALL tuple, or its value
    drifts. A pipeline restart grants a fresh overlay budget, so this
    must RESTART; only if the crash persists does the watchdog's OWN cap
    fire its existing give-up dialog. A STOP_ALL here would kill the
    whole session on the first overlay flap-out. Pins the value: it is
    an inter-process contract.

    Seeded defect: add EXIT_CHILD_GIVEUP to the STOP_ALL tuple in
    classify() -> classify returns STOP_ALL -> red.
    """
    assert EXIT_CHILD_GIVEUP == 47
    assert classify(EXIT_CHILD_GIVEUP) is ExitPolicy.RESTART


def test_exit_code_constants_are_distinct():
    """Fails red if: two sentinels collide (e.g. a new code copy-pasted
    onto an existing value). The watchdog classifies by integer, so a
    collision silently gives one outcome another's policy.

    Seeded defect: set EXIT_CHILD_GIVEUP = 46 (== EXIT_STARTUP_FAILURE)
    -> the set-length assertion goes red.
    """
    codes = [
        EXIT_CLEAN, EXIT_WIZARD_CANCEL, EXIT_TAILORED_ERROR,
        EXIT_STREAM_DEATH, EXIT_STAGE_ESCALATION, EXIT_STARTUP_FAILURE,
        EXIT_CHILD_GIVEUP,
    ]
    assert len(set(codes)) == len(codes), (
        f"exit-code sentinels are not all distinct: {codes}"
    )


def test_exit_code_constants_are_pinned():
    """Fails red if: the canonical sentinel values drift. These integers
    are an INTER-PROCESS contract: the pipeline child raises them, the
    watchdog parent classifies them -- two separate processes, so the
    values themselves (not just the symbols) are load-bearing.

    The cross-file consistency check lives in
    test_main_py_exit_literals_match_constants below.
    """
    assert EXIT_CLEAN == 0
    assert EXIT_WIZARD_CANCEL == 42
    assert EXIT_TAILORED_ERROR == 43
    assert EXIT_STARTUP_FAILURE == 46
    assert EXIT_CHILD_GIVEUP == 47


def test_main_py_exit_literals_match_constants():
    """Fails red if: any hardcoded ``sys.exit(<int>)`` literal in
    yamabiko/main.py drifts from the lifecycle.exit_codes constants.

    Parses main.py's AST (comments and docstrings are invisible, unlike a
    regex) and collects every ``sys.exit()`` call with an integer-literal
    argument, asserting:

      1. every literal is one of the known sentinels
         {EXIT_CLEAN, EXIT_WIZARD_CANCEL, EXIT_TAILORED_ERROR}, and
      2. the tailored-error exit path actually exists (>= 1 site uses
         EXIT_TAILORED_ERROR) -- guarding against the check passing
         vacuously if all literal exits were refactored away without
         this test being updated.

    The wizard-cancel site moved to yamabiko/wizard.py, which imports
    EXIT_WIZARD_CANCEL directly (structurally drift-proof), so main.py
    is not required to contain a 42 literal.

    Seeded-defect drill: edit ``sys.exit(43)`` -> ``sys.exit(44)`` in
    main.py -> the membership assertion goes red naming the line.

    NOTE for maintainers: the structurally-correct fix is for main.py to
    use ``sys.exit(EXIT_TAILORED_ERROR)`` (import the constant); this
    test then keeps guarding any remaining/new literals.
    """
    import ast
    from pathlib import Path

    import yamabiko

    main_path = Path(yamabiko.__file__).resolve().parent / "main.py"
    tree = ast.parse(main_path.read_text(encoding="utf-8"))

    literals: list[tuple[int, int]] = []   # (lineno, value)
    named_sites: list[tuple[int, str]] = []  # ...exit(EXIT_*) by symbol
    # Scan every process-exit delivery path, not just sys.exit: the 43/44 codes
    # actually reach the OS through _hard_exit -> os._exit (os._exit bypasses
    # finalization), so a rogue _hard_exit(99) would slip past a sys.exit-only
    # scan -- the very thing this test exists to stop.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.args):
            continue
        func = node.func
        is_sys_exit = (
            isinstance(func, ast.Attribute) and func.attr == "exit"
            and isinstance(func.value, ast.Name) and func.value.id == "sys"
        )
        is_os_exit = (
            isinstance(func, ast.Attribute) and func.attr == "_exit"
            and isinstance(func.value, ast.Name) and func.value.id == "os"
        )
        is_hard_exit = isinstance(func, ast.Name) and func.id == "_hard_exit"
        if not (is_sys_exit or is_os_exit or is_hard_exit):
            continue
        arg = node.args[0]
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, int)
            and not isinstance(arg.value, bool)
        ):
            literals.append((node.lineno, arg.value))
        elif isinstance(arg, ast.Name) and arg.id.startswith("EXIT_"):
            named_sites.append((node.lineno, arg.id))

    allowed = {
        EXIT_CLEAN, EXIT_WIZARD_CANCEL, EXIT_TAILORED_ERROR, EXIT_STREAM_DEATH,
        EXIT_STAGE_ESCALATION, EXIT_STARTUP_FAILURE, EXIT_CHILD_GIVEUP,
    }
    rogue = [(line, val) for line, val in literals if val not in allowed]
    assert not rogue, (
        f"main.py contains sys.exit() literal(s) outside the "
        f"lifecycle.exit_codes contract {sorted(allowed)}: {rogue} "
        f"(line, value). classify() would treat these as RESTART and "
        f"the watchdog would crash-loop the exit path."
    )
    # The tailored-error exit path must exist -- either as a 43 literal
    # (today's main.py) or, if refactored, as a named
    # sys.exit(EXIT_TAILORED_ERROR) site (the structurally drift-proof
    # form; the membership check above then guards any new literals).
    tailored_sites = [line for line, val in literals if val == EXIT_TAILORED_ERROR]
    tailored_named = [line for line, name in named_sites
                      if name == "EXIT_TAILORED_ERROR"]
    assert tailored_sites or tailored_named, (
        "no sys.exit(43) literal AND no sys.exit(EXIT_TAILORED_ERROR) "
        "site found in main.py -- the tailored-error exit path was "
        "deleted; the watchdog would never see STOP_ALL for startup "
        "errors."
    )
