"""Runtime state shared across pipeline stages.

Mutable. Lock-protected. Accessed by:
  - LoopbackCapture's PortAudio callback (calls `pause_snapshot()` on every
    captured block, from PortAudio's own OS thread)
  - chunker.run (reads the pause snapshot before forwarding frames)
  - translation_stage (reads `genre` per segment; reads `series_media_id`
    per segment to hot-swap the active proper-noun gazetteer)
  - browser_server /ws handler (writes `genre`/`paused` via command
    messages, `series_media_id`/`series_label`/`series_native` via
    `set_series`, and the browser page's reduce motion via `apply_style`)
  - overlay_app tray (writes via command messages from a separate process
    -- arrives over the WebSocket and is applied by the handler above)
  - PipelineApp (problem set, plan phase 4) and the /ws handler (problems frame)

The PortAudio callback thread reads the pause state concurrently with the
asyncio loop that writes it, so the lock is required: it keeps every
epoch/level pair atomic for that foreign thread.

Keep this module dep-free so tests and the pipeline can import it without
pulling in Qt or FastAPI.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from numbers import Integral

from .overlay_app.style_defaults import STYLE_DEFAULTS
from .problems import make_problems_frame, ordered_kinds

VALID_GENRES: tuple[str, ...] = ("variety", "drama", "news", "anime")


@dataclass(frozen=True, slots=True)
class PauseState:
    """One atomic capture-facing view of the pause boundary."""

    epoch: int
    paused: bool

    def __post_init__(self) -> None:
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, Integral):
            raise TypeError("epoch must be a non-negative integer")
        if self.epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if not isinstance(self.paused, bool):
            raise TypeError("paused must be a bool")
        object.__setattr__(self, "epoch", int(self.epoch))


@dataclass
class AppState:
    genre: str = "variety"
    paused: bool = False
    # Active series for the proper-noun gazetteer, chosen via the series
    # palette. media_id is the AniList id; label is the human string the
    # tray shows ("Series (year)"). Both None until a series is picked.
    series_media_id: int | None = None
    series_label: str | None = None
    # The picked series' Japanese (AniList native) title, for the overlay
    # banner. None when unknown (no pick, or a pointer saved before it).
    series_native: str | None = None
    # Bumped on EVERY set_series call, including a re-pick of the same
    # media_id: the pipeline hot loop reloads on epoch change, so a
    # freshly built cache for the already-active series still loads.
    # Deliberately NOT part of state_frame()/snapshot() -- the wire
    # protocol is unchanged; the hot loop reads this field directly.
    series_epoch: int = 0
    # User-facing {"type": "notice", "title", "body"} frames raised while the
    # Qt overlay (the native WS client, which sends no Origin) wasn't
    # connected (a hotkey conflict fires during startup, before the server
    # exists). The /ws handler sends and clears them right after the next
    # native client's problems frame; browser clients leave them queued.
    # Bounded so a client that never connects cannot grow it.
    pending_notices: deque = field(
        default_factory=lambda: deque(maxlen=8), repr=False, compare=False,
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _style: dict = field(default_factory=lambda: dict(STYLE_DEFAULTS), repr=False, compare=False)
    # overlay.reduce_motion, for the browser page (the style frame carries
    # it). Seeded by main.build_pipeline; updated by the tray's apply_style.
    _reduce_motion: bool = field(default=False, repr=False, compare=False)
    # asr.incremental.enabled ("Faster subtitles"). Seeded by
    # main.build_pipeline; changed live by the tray's set_incremental; the ASR
    # stage reads it when it first sees a window.
    _incremental: bool = field(default=False, repr=False, compare=False)
    # Active problems by kind (problems.py builds each entry: kind, title,
    # body, line, recovery), sent as the "problems" frame.
    _problems: dict = field(default_factory=dict, repr=False, compare=False)
    # Kinds the previous pipeline restarted on and this one resolved before
    # it accepted clients (problems.CARRY_ENV), and the monotonic deadline
    # until which a new client's connect frame lists them as recovered.
    _carried: tuple = field(default=(), repr=False, compare=False)
    _carried_until: float = field(default=0.0, repr=False, compare=False)
    _pause_state: PauseState = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._pause_state = PauseState(epoch=0, paused=bool(self.paused))
        self.paused = self._pause_state.paused
        # Validate at construction. main.py builds AppState(genre=cfg["genre"])
        # directly from the merged config (bypassing set_genre), and users
        # hand-edit user_overrides.yaml -- a typo like `genre: anme` used to
        # flow through unchecked and silently drop the genre-steering prompt
        # note (GENRE_NOTES.get(genre, "")). Coerce + warn instead.
        if self.genre not in VALID_GENRES:
            import logging
            logging.getLogger(__name__).warning(
                "invalid genre %r; falling back to 'variety' (valid: %s)",
                self.genre, ", ".join(VALID_GENRES),
            )
            self.genre = "variety"

    def set_genre(self, genre: str) -> None:
        if genre not in VALID_GENRES:
            raise ValueError(f"invalid genre: {genre!r}; must be one of {VALID_GENRES}")
        with self._lock:
            self.genre = genre

    def pause_snapshot(self) -> PauseState:
        """Return one immutable epoch/level pair without a torn read."""

        with self._lock:
            return self._pause_state

    def set_paused(self, paused: bool) -> PauseState | None:
        """Apply one real pause edge; repeated same-state commands are no-ops."""

        target = bool(paused)
        with self._lock:
            current = self._pause_state
            if current.paused is target:
                return None
            edge = PauseState(epoch=current.epoch + 1, paused=target)
            self._pause_state = edge
            # Keep the long-standing public level and wire schema compatible.
            self.paused = edge.paused
            return edge

    def pause(self) -> PauseState | None:
        return self.set_paused(True)

    def resume(self) -> PauseState | None:
        return self.set_paused(False)

    def set_series(
        self, media_id: int | None, label: str | None, native: str | None = None,
    ) -> None:
        with self._lock:
            self.series_media_id = media_id
            self.series_label = label
            self.series_native = native
            # Every call is a change event, even a re-pick of the same id
            # (see series_epoch above): the first pick may have recorded an
            # empty AniList result, and the re-pick must reload the cache
            # built since. The hot loop gates on this, not on id inequality.
            self.series_epoch += 1

    def queue_notice(self, notice: dict) -> None:
        """Hold one notice frame for the next native (no-Origin) WS client,
        the Qt overlay (oldest dropped past 8)."""
        with self._lock:
            self.pending_notices.append(dict(notice))

    def take_pending_notices(self) -> list[dict]:
        """Return and clear every queued notice, oldest first."""
        with self._lock:
            notices = list(self.pending_notices)
            self.pending_notices.clear()
            return notices

    def set_problem(self, problem: dict) -> bool:
        """Make ``problem`` the active entry of its kind; True when the set
        changed (a repeat of the same entry is not a change)."""
        with self._lock:
            if self._problems.get(problem["kind"]) == problem:
                return False
            self._problems[problem["kind"]] = dict(problem)
            return True

    def clear_problem(self, kind: str) -> bool:
        """Drop ``kind``; True when it was active."""
        with self._lock:
            return self._problems.pop(kind, None) is not None

    def has_problem(self, kind: str) -> bool:
        with self._lock:
            return kind in self._problems

    def problem_kinds(self) -> tuple[str, ...]:
        with self._lock:
            return ordered_kinds(self._problems)

    def set_carried_recoveries(self, kinds, until: float) -> None:
        with self._lock:
            self._carried = ordered_kinds(kinds)
            self._carried_until = float(until)

    def problems_frame(self, *, now: float | None = None) -> dict:
        """WS-ready ``{"type": "problems", ...}``. Only a new connection
        passes ``now`` (a monotonic time): inside the carried window its
        frame also lists the carried recoveries."""
        with self._lock:
            active = list(self._problems.values())
            carried = (
                self._carried
                if now is not None and now < self._carried_until
                else ()
            )
        return make_problems_frame(active, carried)

    def set_style(self, style: dict) -> None:
        """Merge ``style`` into the stored style dict (no broadcast payload).

        Used on initial load from the on-disk config. Callers that want to
        notify connected clients should use :meth:`update_style` instead.
        """
        with self._lock:
            self._style.update(style)

    def set_reduce_motion(self, on: bool) -> None:
        with self._lock:
            self._reduce_motion = bool(on)

    def set_incremental(self, on: bool) -> None:
        with self._lock:
            self._incremental = bool(on)

    def incremental_enabled(self) -> bool:
        with self._lock:
            return self._incremental

    def update_style(self, changes: dict) -> dict:
        """Merge ``changes`` and return the WS-ready style frame."""
        with self._lock:
            self._style.update(changes)
            return self._style_frame_locked()

    def style_frame(self) -> dict:
        """WS-ready ``{"type": "style", ...}``: the full style plus
        ``reduce_motion``. THE builder for the on-connect and change frames."""
        with self._lock:
            return self._style_frame_locked()

    def _style_frame_locked(self) -> dict:
        return {"type": "style", **self._style, "reduce_motion": self._reduce_motion}

    def snapshot(self) -> dict:
        """Return a JSON-safe snapshot for broadcasting state updates."""
        with self._lock:
            return {
                "genre": self.genre,
                "paused": self.paused,
                "series_media_id": self.series_media_id,
                "series_label": self.series_label,
                "series_native": self.series_native,
                "style": dict(self._style),
            }

    def state_frame(self) -> dict:
        """WS-ready ``{"type": "state", ...}`` frame -- THE single builder
        every state broadcast routes through (the command handlers, the
        on-connect snapshot, and the pause hotkey).

        The series keys are always present here on purpose: ws_client only
        touches the tray's Series line for a frame that CARRIES the keys, so
        a producer that hand-builds a frame omitting them makes the client
        read the series as cleared. Building the frame in one place keeps
        every producer honest.
        """
        snap = self.snapshot()
        return {
            "type": "state",
            "genre": snap["genre"],
            "paused": snap["paused"],
            "series_media_id": snap["series_media_id"],
            "series_label": snap["series_label"],
            "series_native": snap["series_native"],
        }
