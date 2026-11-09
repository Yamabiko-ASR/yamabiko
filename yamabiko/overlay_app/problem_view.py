"""ProblemTracker: the overlay process's view of the pipeline's problems.

The pipeline sends its whole problem set as a ``problems`` frame on connect
and on every change (yamabiko/problems.py, PipelineClient.problemsReceived).
The tracker keeps the last set it saw for the life of the process, across
reconnects, and turns each frame into what changed (plan Decisions 1, 18,
22 and 43):

* ``changed(active)``: the set differs from the last one (a kind came or
  went, or an entry's text changed). The overlay redraws its problem lines;
  the tray its dot, tooltip and flyout status.
* ``recovered(kind, text)``: a kind left the set, or a restarted pipeline
  reports one it resolved (the frame's ``recovered`` list; shown once per
  kind per process). The overlay shows the recovery line.
* ``started(entry)``: a kind is new in the set. The tray shows its toast.

Per frame they fire in that order, so the overlay drops a problem line
before it adds the recovery line of the same kind.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from ..problems import KIND_ORDER, is_problem


class ProblemTracker(QObject):
    changed = Signal(list)          # the active entries, display order
    recovered = Signal(str, str)    # kind, recovery text
    started = Signal(dict)          # the new entry

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._active: dict[str, dict] = {}
        self._carried_shown: set[str] = set()

    def active(self) -> list[dict]:
        """Copies of the active entries, translator before audio."""
        return [dict(self._active[kind]) for kind in KIND_ORDER if kind in self._active]

    def apply(self, active: list, recovered: list) -> None:
        """One ``problems`` frame: the ``active`` entries and the carried
        ``recovered`` entries (``{"kind", "recovery"}``)."""
        new = {entry["kind"]: dict(entry) for entry in active if is_problem(entry)}
        old = self._active
        gone = [kind for kind in KIND_ORDER if kind in old and kind not in new]
        came = [kind for kind in KIND_ORDER if kind in new and kind not in old]
        carried: dict[str, str] = {}
        for entry in recovered:
            if not isinstance(entry, dict):
                continue
            kind, text = entry.get("kind"), entry.get("recovery")
            if (kind in KIND_ORDER and isinstance(text, str) and kind not in new
                    and kind not in gone and kind not in self._carried_shown):
                carried[kind] = text
        self._active = new
        self._carried_shown.update(carried)
        if new != old:
            self.changed.emit(self.active())
        for kind in KIND_ORDER:
            if kind in gone:
                self.recovered.emit(kind, old[kind]["recovery"])
            elif kind in carried:
                self.recovered.emit(kind, carried[kind])
        for kind in came:
            self.started.emit(dict(new[kind]))
