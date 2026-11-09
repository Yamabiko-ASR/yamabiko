"""Launch-chain log markers — Qt-free constants the launch-chain tests assert on.

Kept here (not in ``wizard.py``) so the marker can be imported without dragging
the PySide6 stack: the frozen-bundle E2E module (``tests/test_real_frozen_e2e``)
is deliberately Qt-free at module scope, and the launch-chain tests need only the
string, not the wizard. Pairs with this package's ``exit_codes`` — the same
tests assert ``EXIT_WIZARD_CANCEL == 42`` alongside this marker.
"""
from __future__ import annotations

# ``run_wizard()`` logs this as its FIRST statement (issue #25). It lands in
# ``<appdata>/logs/main-pipeline.log`` and is the ONLY observable that proves
# the Start-Menu -> watchdog ``--run-main`` -> ``main.main()`` ->
# ``needs_wizard()`` gate actually reached ``run_wizard()``: exit 42 alone is
# reachable WITHOUT it (a fall-through gate still leaves needs_wizard() True ->
# ``exit_wizard_cancelled()`` -> 42), so the marker is the launch-chain tests'
# anti-Liar observable.
WIZARD_START_MARKER = "first-run wizard starting (wizard_state.needs_wizard() was True)"
