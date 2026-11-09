"""Centralised exit-code policy for watchdog-supervised children.

Single source of truth. Adding a new fatal code = one line here, no
touches to watchdog.py. This prevents the exit-code policy from drifting
across modules.
"""
from __future__ import annotations

from enum import Enum, auto

EXIT_CLEAN = 0
EXIT_WIZARD_CANCEL = 42
EXIT_TAILORED_ERROR = 43
# Capture stream died unexpectedly (device unplug / suspended / active-but-
# silent). MUST stay OUT of the STOP_ALL set below so classify() routes it to
# RESTART -- the watchdog rebuilds the pipeline and re-resolves the device.
# Delivered to the watchdog via main()'s _hard_exit (os._exit), so native-
# extension-thread finalization can't mangle the code en route.
EXIT_STREAM_DEATH = 44
# A pipeline stage crossed its supervisor escalation threshold (repeated
# crashes, e.g. translation failing 10x in 60 s) and took the pipeline down
# deliberately. MUST stay OUT of the STOP_ALL set below so classify() routes it
# to RESTART -- the watchdog rebuilds the pipeline; if the failure persists,
# the restart's startup sanity check exits EXIT_TAILORED_ERROR with a dialog,
# or repeated escalations exhaust the anti-flap budget into the give-up dialog.
# (Escalation used to exit 0 = STOP_SELF: the watchdog read a stage crash-loop
# as a clean quit and the app silently died with no dialog and no restart.)
# Delivered to the watchdog via main()'s _hard_exit (os._exit), same unmangled
# channel as EXIT_STREAM_DEATH.
EXIT_STAGE_ESCALATION = 45
# The overlay HTTP/WebSocket server could not bind its port at startup
# (another program holds overlay.websocket_port, or a hard-killed prior
# run left a lingering socket). MUST stay IN the STOP_ALL set below so
# classify() routes it to a terminal cascade: a bind conflict does not
# self-heal in seconds, so restart churn just delays the message. The
# watchdog additionally surfaces a port-specific give-up dialog for this
# code. (Bind failure used to reach _exit_code_for with no armed flag =>
# exit 0 = STOP_SELF: the watchdog read the failed startup as a clean
# quit and the whole app silently disappeared.) Delivered to the watchdog
# via main()'s _hard_exit (os._exit), same unmangled channel as the codes
# above.
EXIT_STARTUP_FAILURE = 46
# The pipeline's overlay supervisor exhausted its respawn budget (the
# overlay crash-looped past the anti-flap cap). MUST stay OUT of the
# STOP_ALL set so classify() routes it to RESTART: a pipeline restart
# grants a fresh overlay budget, and only if the crash is persistent does
# the watchdog's OWN 3-in-window cap then fire its existing give-up
# dialog. (Give-up used to set the same overlay_stop event as a normal
# overlay exit => _exit_code_for saw no reason => exit 0 = STOP_SELF: a
# flapped-out overlay was indistinguishable from a user quit.) Delivered
# via _hard_exit, same channel as EXIT_STREAM_DEATH.
EXIT_CHILD_GIVEUP = 47

# Not one of OUR codes: the Windows NTSTATUS a process exits with when a console
# control signal kills it (Ctrl+C, or its console window being closed).
# GetExitCodeProcess reports it as the unsigned DWORD, and CPython's
# _winapi.GetExitCodeProcess marshals that with Py_BuildValue("k"), so
# Popen.returncode carries the POSITIVE 3221225786 (field-confirmed: a real
# watchdog log records "raw code=3221225786"). Do NOT rewrite this as the signed
# twin (-1073741510); the comparison would never fire.
#
# Deliberately absent from classify(): a child killed from outside is still a
# RESTART candidate by policy (the user may not have meant to kill it). The code
# exists so the give-up SURFACES can tell "somebody stopped this" apart from
# "this is misconfigured" and "this is crash-looping" -- see is_external_stop.
STATUS_CONTROL_C_EXIT = 0xC000013A


def is_external_stop(code: int) -> bool:
    """True when ``code`` means the child was terminated from outside Yamabiko.

    Policy is unchanged (classify() still routes it to RESTART); this only
    tells a give-up dialog not to blame the user's configuration or claim
    Yamabiko is crashing. Lives here so every give-up site can consult one
    definition instead of the supervision loop owning the knowledge.
    """
    return code == STATUS_CONTROL_C_EXIT


class ExitPolicy(Enum):
    RESTART = auto()    # transient crash; supervisor retries per anti-flap
    STOP_ALL = auto()   # unrecoverable; watchdog tears down peers too
    STOP_SELF = auto()  # clean exit; supervisor stops without restart


def classify(code: int) -> ExitPolicy:
    if code == EXIT_CLEAN:
        return ExitPolicy.STOP_SELF
    if code in (EXIT_WIZARD_CANCEL, EXIT_TAILORED_ERROR, EXIT_STARTUP_FAILURE):
        return ExitPolicy.STOP_ALL
    return ExitPolicy.RESTART
