"""Minimum-edit character alignment, shared by the scorer and the island
second pass."""

from __future__ import annotations

import numpy as np


def alignment_trace(reference: str, hypothesis: str) -> tuple[np.ndarray, np.ndarray, int]:
    """One minimum-edit global alignment of ``reference`` with ``hypothesis``.

    Returns, per ``reference`` character, whether it aligns as an exact match
    and its hypothesis position: the index of the hypothesis character it
    aligns with, or for a deleted character the index it falls before; and
    the edit distance.

    The DP runs on the reversed strings, so ``cost[a, b]`` is the distance
    between the last ``a`` reference and last ``b`` hypothesis characters;
    the traceback then walks forward from the start and prefers match,
    substitution, deletion, insertion, the E0 ``character_trace`` order.
    Each row is vectorized: substitution and deletion elementwise, then the
    insertion chain as a running minimum of ``row - j`` plus ``j``.
    """
    n, m = len(reference), len(hypothesis)
    matched = np.zeros(n, dtype=bool)
    positions = np.full(n, m, dtype=np.int64)
    if n == 0 or m == 0:
        return matched, positions, n + m
    ref = np.frombuffer(reference[::-1].encode("utf-32-le"), dtype=np.uint32)
    hyp = np.frombuffer(hypothesis[::-1].encode("utf-32-le"), dtype=np.uint32)
    dtype = np.int16 if n + m < 2**15 else np.int32
    cost = np.empty((n + 1, m + 1), dtype=dtype)
    steps = np.arange(m + 1, dtype=dtype)
    cost[0] = steps
    row = np.empty(m + 1, dtype=dtype)
    for i in range(1, n + 1):
        previous = cost[i - 1]
        row[0] = i
        np.minimum(previous[:-1] + (hyp != ref[i - 1]), previous[1:] + 1, out=row[1:])
        row -= steps
        np.minimum.accumulate(row, out=cost[i])
        cost[i] += steps
    a, b = n, m
    while a and b:
        positions[n - a] = m - b
        # Equal characters always admit the match step at no extra cost.
        if reference[n - a] == hypothesis[m - b]:
            matched[n - a] = True
            a -= 1
            b -= 1
        elif cost[a, b] == cost[a - 1, b - 1] + 1:
            a -= 1
            b -= 1
        elif cost[a, b] == cost[a - 1, b] + 1:
            a -= 1
        else:
            b -= 1
    return matched, positions, int(cost[n, m])
