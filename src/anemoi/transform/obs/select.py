# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Per-cell candidates and their reduction to one value per cell: sorts and run starts,
no groupby, so a day of IASI (millions of rows) takes seconds."""

import numpy as np


def _run_starts(sorted_keys: np.ndarray) -> np.ndarray:
    """Mask of the first element of each run of equal keys."""
    return np.r_[True, sorted_keys[1:] != sorted_keys[:-1]] if sorted_keys.size else sorted_keys


def candidate_groups(
    cell: np.ndarray,
    t: np.ndarray,
    dist: np.ndarray,
    candidates: str,
    pass_gap: int,
    group: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(order, starts): the indices of each cell's candidates, cell by cell with the latest
    first (ties: the nearer), and where each cell's run starts in `order`.

    candidates: "latest_pass" keeps the cell's rows within pass_gap (t's units) of its
    latest row, "window" keeps them all. group (e.g. satid): keep only rows of the latest
    row's group. The latest row always stays, so every cell with a row has candidates.
    """
    order = np.lexsort((dist, -t, cell))
    first = _run_starts(cell[order])
    if candidates not in ("latest_pass", "window"):
        raise ValueError(f"unknown candidates {candidates!r}")
    if order.size and (candidates == "latest_pass" or group is not None):
        run = np.cumsum(first) - 1
        keep = np.ones(order.size, bool)
        if candidates == "latest_pass":
            keep &= t[order] >= t[order][first][run] - pass_gap
        if group is not None:
            keep &= group[order] == group[order][first][run]
        order = order[keep]
    return order, np.flatnonzero(_run_starts(cell[order]))


def select_cells(
    cell: np.ndarray,
    t: np.ndarray,
    dist: np.ndarray,
    candidates: str,
    reduce: str,
    pass_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per cell: (cells, row index into the inputs, number of candidates).

    reduce: "nearest" takes the candidate nearest the cell centre (ties: the later),
    "latest" the latest (ties: the nearer). Remaining ties keep input order, so the pick is
    deterministic.
    """
    order, starts = candidate_groups(cell, t, dist, candidates, pass_gap)
    if reduce == "nearest":
        order = nearest_first(order, cell, t, dist)
    elif reduce != "latest":
        raise ValueError(f"select_cells picks a row: unknown reduce {reduce!r}")
    rows = order[starts]
    return cell[rows], rows, np.diff(np.r_[starts, order.size])


def nearest_first(order: np.ndarray, cell: np.ndarray, t: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """The same runs of candidate_groups, the nearest first within each cell (ties: the later)."""
    return order[np.lexsort((-t[order], dist[order], cell[order]))]


def group_mean(x: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Mean of the finite values of each run of x (runs start at `starts`); NaN if none."""
    total, count = _group_sums(x, starts, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total[0] / count, np.nan)


def group_std(x: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Population standard deviation of the finite values of each run; NaN if none."""
    (total, total_sq), count = _group_sums(x, starts, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        return np.where(count > 0, np.sqrt(np.maximum(total_sq / count - mean**2, 0)), np.nan)


def _group_sums(x: np.ndarray, starts: np.ndarray, powers: int):
    if starts.size == 0:
        return np.zeros((powers, 0)), np.zeros(0)
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0).astype(np.float64)
    sums = np.stack([np.add.reduceat(v**k, starts) for k in range(1, powers + 1)])
    return sums, np.add.reduceat(ok.astype(np.int64), starts)
