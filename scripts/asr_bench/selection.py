"""Which episodes of a title and which windows of an episode the episode silver
set takes.

``build_silver_episodes`` takes, per title, ``spaced`` episodes of the
title's eligible episodes in ``sort_key`` order, and from each episode the
``windows`` spans. The inventory's bounded decode check and the external
subtitle builder use ``walk`` to check only the episodes the builder can
take: the same evenly spaced picks, each falling to its nearest untried
neighbour when it fails, so the cost stays bounded by the episodes needed,
not the episodes on disk.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

WINDOW_S = 240.0
EDGE_SKIP_S = 120.0
DEFAULT_WINDOWS = 2

Record = TypeVar("Record")


def window_centres(count: int) -> list[float]:
    """Window centres as fractions of the usable span: 0.5 for one window,
    else evenly spaced from 0.3 to 0.7 (``[0.3, 0.7]`` for two)."""
    if count < 1:
        raise ValueError("an episode needs at least one window")
    if count == 1:
        return [0.5]
    return [round(0.3 + 0.4 * index / (count - 1), 6) for index in range(count)]


def windows(duration_s: float, count: int = DEFAULT_WINDOWS) -> list[tuple[float, float]] | None:
    """The window spans of an episode, or None when they do not fit inside
    [EDGE_SKIP_S, duration - EDGE_SKIP_S] without overlapping."""
    low, high = EDGE_SKIP_S, duration_s - EDGE_SKIP_S
    half = WINDOW_S / 2
    spans = [(low + f * (high - low) - half, low + f * (high - low) + half) for f in window_centres(count)]
    fits = spans[0][0] >= low and spans[-1][1] <= high
    disjoint = all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
    return spans if fits and disjoint else None


def spaced(count: int, limit: int) -> list[int]:
    """``limit`` indices evenly spaced through ``count`` (all when fewer)."""
    if count <= limit:
        return list(range(count))
    return [(2 * index + 1) * count // (2 * limit) for index in range(limit)]


def sort_key(record: Mapping) -> tuple:
    """Season (none first), episode number (none last), file name."""
    episode = record["episode"]
    return record["season"] or 0, episode if episode is not None else math.inf, record["file"]


def walk(
    candidates: Sequence[Record], limit: int, accept: Callable[[Record], bool], max_tries: int | None = None
) -> list[Record]:
    """Up to ``limit`` candidates ``accept`` passes, in candidate order.

    Slot i starts at ``spaced(len, limit)[i]``. In rounds, every slot still
    empty tries its nearest untried candidate (the later one on a tie), so a
    failing pick falls to its neighbours and, once they are used up, to the
    nearest candidates beyond them. ``max_tries`` bounds the calls to
    ``accept``; each candidate is tried at most once.
    """
    count = len(candidates)
    picks = spaced(count, limit)
    tried: set[int] = set()
    chosen: dict[int, int] = {}
    tries = 0
    while len(chosen) < len(picks) and len(tried) < count:
        progressed = False
        for slot, pick in enumerate(picks):
            if slot in chosen:
                continue
            untried = [j for j in range(count) if j not in tried]
            if not untried:
                break
            index = min(untried, key=lambda j: (abs(j - pick), -j))
            tried.add(index)
            tries += 1
            progressed = True
            if accept(candidates[index]):
                chosen[slot] = index
            if max_tries is not None and tries >= max_tries:
                return [candidates[j] for j in sorted(chosen.values())]
        if not progressed:
            break
    return [candidates[j] for j in sorted(chosen.values())]
