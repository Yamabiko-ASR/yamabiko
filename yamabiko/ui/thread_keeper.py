"""Registry for QThreads that must outlive their owning widget.

Qt 6 calls ``qFatal("QThread: Destroyed while thread is still running")``
— an abort(), not an exception — if a ``QThread`` C++ object is deleted
while its OS thread runs. The wizard's download worker can be blocked
inside ``hf_hub_download`` for minutes (the 5.5 GB GGUF), and that call
cannot be interrupted in-process: the xet transfer path swallows
exceptions raised from the tqdm progress callback (huggingface_hub +
hf_xet). So when the wizard is closed mid-download, the only safe move
is to *detach* the thread (strip its widget parent and park a Python
reference here) instead of letting widget teardown destroy it.

Entries are never pruned by production code (tests prune via
:func:`discard`): a wizard run parks at most two threads (downloads +
smoke probe), once per process lifetime, and a *finished* QThread is
safe to destroy at interpreter shutdown. The deliberate leak is bounded
and harmless; pruning from cross-thread ``finished`` signals is not (no
Qt event loop is guaranteed to be running after the wizard closes).

``main.py``'s wizard-cancel path consults :func:`any_running`: if a
detached thread is still draining, it must hard-exit via ``os._exit``
so interpreter finalization can't destroy the running thread (and so
the exit-42 wizard-cancel sentinel reaches the watchdog intact —
anything else is classified RESTART and relaunches the wizard the user
just cancelled).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from PySide6.QtCore import QObject, QThread

_log = logging.getLogger(__name__)

_LINGERING: List[Tuple[QThread, Optional[QObject]]] = []


def release(thread: QThread, worker: Optional[QObject] = None) -> None:
    """Detach ``thread`` from its parent and keep it (plus its worker)
    alive for the rest of the process.

    The worker reference matters too: it is the QObject executing on the
    thread; dropping its last Python reference mid-run would let PySide
    delete it under the running code.
    """
    thread.setParent(None)
    _LINGERING.append((thread, worker))
    _log.info(
        "detached running background thread (%d lingering total)",
        len(_LINGERING),
    )


def any_running() -> bool:
    """True if any detached thread is still executing."""
    return any(t.isRunning() for t, _ in _LINGERING)


def discard(thread: QThread) -> None:
    """Drop ``thread``'s registry entry.

    Test-teardown surface: production never prunes (see module
    docstring), but tests that deliberately release a blocked thread
    must remove it after joining so the global registry can't leak
    state into later tests.
    """
    _LINGERING[:] = [(t, w) for t, w in _LINGERING if t is not thread]
