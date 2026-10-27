# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Observation rows → one value per (window, grid cell): QC → candidates per cell →
nearest / latest / mean → derived variables, plus forcings at (T, grid point).

A window is stamped T and holds the rows with T + window.before < date <= T + window.after
(brackets as the anemoi.utils Window string, e.g. "(-350m,+10m]").
"""

import contextlib
import datetime
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import pandas as pd
from anemoi.utils.window import Window

from anemoi.transform.obs import derived
from anemoi.transform.obs import grid as grids
from anemoi.transform.obs import qc
from anemoi.transform.obs import select
from anemoi.transform.obs.config import Gridding


@dataclass
class GriddedWindows:
    """What grid_windows returns.

    arrays: {output name: float32 (window, grid point)}, NaN where no obs was selected.
    picked: the selected values of every window, with `step` (T) and the group column.
    qc: QC counts (group, step, n) of the input rows (anemoi.transform.obs.qc.apply).
    sums: per grid point, cover (windows with a value), maps (n_maps, [count, sum, sum_sq])
    and qc ([rows, rejected]).
    """

    arrays: dict[str, np.ndarray]
    picked: pd.DataFrame
    qc: pd.DataFrame
    sums: dict[str, np.ndarray] = field(default_factory=dict)

    def stats(self, columns: list[str], by: str) -> pd.DataFrame:
        """Per (day of T, group, column) count, nan_count, sum, sum_sq, min and max of the
        selected values: sums merge exactly, so any period's mean and std follow."""
        if self.picked.empty:
            return pd.DataFrame(columns=["day", "group", "column", "count", "nan_count", "sum", "sum_sq", "min", "max"])
        keys = [self.picked["step"].dt.floor("D").rename("day"), self.picked[by].rename("group")]
        rows = []
        for col in columns:
            values = self.picked[col].astype(np.float64)
            g = values.groupby(keys)
            rows.append(
                pd.DataFrame(
                    {
                        "column": col,
                        "count": g.count(),
                        "nan_count": g.size() - g.count(),
                        "sum": g.sum(),
                        "sum_sq": (values**2).groupby(keys).sum(),
                        "min": g.min(),
                        "max": g.max(),
                    }
                ).reset_index()
            )
        return pd.concat(rows, ignore_index=True)


def window_bounds(window: Window, time: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(lo, hi): the bounds of the window stamped `time`."""
    return time + pd.Timedelta(window.before), time + pd.Timedelta(window.after)


def in_window(t: np.ndarray, window: Window, time: pd.Timestamp) -> np.ndarray:
    """Mask of the int64 ns times t inside the window stamped `time`."""
    lo, hi = window_bounds(window, time)
    after_lo = t > lo.value if window.exclude_before else t >= lo.value
    before_hi = t < hi.value if window.exclude_after else t <= hi.value
    return after_lo & before_hi


def read_interval(window: Window, steps: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end] covering every window of `steps` (an input source reads this)."""
    return window_bounds(window, steps[0])[0], window_bounds(window, steps[-1])[1]


def row_values(
    cfg: Gridding,
    df: pd.DataFrame,
    rows: np.ndarray,
    time: pd.Timestamp,
    freq: datetime.timedelta,
    east: np.ndarray,
    north: np.ndarray,
) -> dict[str, np.ndarray]:
    """{output name: float32 per row} of every observed variable but n_obs and spreads."""
    v = cfg.variables
    pick = df.iloc[rows]
    out = {v.out_name(c): pick[c].to_numpy(dtype=np.float32, na_value=np.nan) for c in (*v.columns, *v.channels)}
    hours = (pick["date"] - time).to_numpy() / np.timedelta64(1, "h")
    freq_h = pd.Timedelta(freq) / pd.Timedelta("1h")
    spacing = grids.mean_spacing_km(cfg.grid)
    scan_pos = None
    if v.scan is not None:
        scan_pos = derived.scan_position(pick[v.scan.column].to_numpy(np.int64), v.scan)
    zenith = pick["zenith"].to_numpy(np.float32) if "zenith" in pick else None
    every = {
        "time_delta": lambda: hours,
        "time_delta_norm": lambda: hours / freq_h,
        "cos_vza": lambda: np.cos(np.radians(zenith)),
        "scan_pos": lambda: scan_pos,
        "signed_sin_vza": lambda: derived.signed_sin_vza(scan_pos, zenith),
        "offset_east_km": lambda: east[rows],
        "offset_north_km": lambda: north[rows],
        "offset_east_norm": lambda: east[rows] / spacing,
        "offset_north_norm": lambda: north[rows] / spacing,
        "cos_sza_obs": lambda: np.cos(np.radians(pick["solar_zenith"].to_numpy(np.float64))),
    }
    names = [n for n in v.derived if n != "n_obs"]
    return out | {n: np.asarray(every[n](), np.float32) for n in names}


def cell_values(
    cfg: Gridding,
    df: pd.DataFrame,
    idx: np.ndarray,
    time: pd.Timestamp,
    freq: datetime.timedelta,
    cell: np.ndarray,
    t: np.ndarray,
    dist: np.ndarray,
    group: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """(cells, {output name: float32 per cell}, group per cell) from one window's rows idx:
    the candidates of each cell (select.candidate_groups), reduced to one value per cell."""
    s, v = cfg.select, cfg.variables
    gap = pd.Timedelta(s.pass_gap).value
    same = group[idx] if s.reduce == "mean" else None  # a superob never mixes groups
    order, starts = select.candidate_groups(cell[idx], t[idx], dist[idx], s.candidates, gap, same)
    if s.reduce == "nearest":
        order = select.nearest_first(order, cell[idx], t[idx], dist[idx])
    rows = idx[order]
    first = rows[starts]
    n_obs = np.diff(np.r_[starts, rows.size])
    if s.reduce == "mean":
        values = row_values(cfg, df, rows, time, freq, east, north)
        radiance = {}
        if s.mean_of == "radiance":
            radiance = {v.out_name(c): v.wavenumbers.of(c) for c in v.channels}
        for name, wavenumber in radiance.items():
            values[name] = derived.bt_to_radiance(values[name], wavenumber)
        out = {k: select.group_mean(x, starts) for k, x in values.items()}
        for name, wavenumber in radiance.items():
            out[name] = derived.radiance_to_bt(out[name], wavenumber)
    else:
        out = row_values(cfg, df, first, time, freq, east, north)
    for c in v.spread:
        x = df[c].to_numpy(dtype=np.float64, na_value=np.nan)[rows]
        out[f"{v.out_name(c)}_std"] = select.group_std(x, starts)
    if "n_obs" in v.derived:
        out["n_obs"] = n_obs
    keep = n_obs >= s.min_obs
    out = {k: np.asarray(x, np.float32)[keep] for k, x in out.items()}
    return cell[first][keep], out, group[first][keep]


def _copy_on_write():
    """pandas < 3 copies the whole input frame on every assign / reset_index (+1 GB per copy
    for a day of IASI); copy-on-write, the pandas 3 default, copies only what changes."""
    if int(pd.__version__.split(".")[0]) >= 3:
        return contextlib.nullcontext()
    return pd.option_context("mode.copy_on_write", True)


def grid_windows(
    cfg: Gridding,
    df: pd.DataFrame,
    steps: pd.DatetimeIndex,
    window: Window | str,
    freq: datetime.timedelta,
) -> GriddedWindows:
    """The windows stamped `steps` from the rows covering them (read_interval). No I/O.

    freq is the dataset frequency (time_delta_norm = time_delta / freq).
    """
    with _copy_on_write():
        return _grid_windows(cfg, df, steps, window, freq)


def _grid_windows(cfg, df, steps, window, freq) -> GriddedWindows:
    if isinstance(window, str):
        window = Window(window)
    v = cfg.variables
    steps = pd.DatetimeIndex(steps)
    lat, lon = grids.grid_points(cfg.grid)
    n = lat.size
    arrays = {name: np.full((len(steps), n), np.nan, np.float32) for name in v.observed()}
    for name in v.forcings:
        arrays[name] = np.empty((len(steps), n), np.float32)
    for i, time in enumerate(steps):
        for name, x in derived.forcings(v.forcings, time, lat, lon).items():
            arrays[name][i] = x
    sums = {"cover": np.zeros(n), "maps": np.zeros((len(cfg.maps), 3, n)), "qc": np.zeros((2, n))}
    if df.empty:
        return _final(GriddedWindows(arrays, pd.DataFrame(), pd.DataFrame(columns=["group", "step", "n"]), sums), cfg)
    df, keep, counts = qc.apply(cfg.qc, df.reset_index(drop=True), v.channels, cfg.group_by)
    obs_lat, obs_lon = df["latitude"].to_numpy(np.float64), df["longitude"].to_numpy(np.float64)
    cell = grids.cell_index(cfg.grid, obs_lat, obs_lon)
    dist, east, north = grids.cell_offsets_km(cfg.grid, obs_lat, obs_lon, cell)
    t = df["date"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    group = df[cfg.group_by].to_numpy(np.int64)
    sums["qc"][0] = np.bincount(cell, minlength=n)
    sums["qc"][1] = np.bincount(cell[~keep], minlength=n)
    picked = []
    for i, time in enumerate(steps):
        idx = np.flatnonzero(keep & in_window(t, window, time))
        cells, values, cell_group = cell_values(cfg, df, idx, time, freq, cell, t, dist, group, east, north)
        for name, x in values.items():
            arrays[name][i, cells] = x
        sums["cover"][cells] += 1
        for k, name in enumerate(cfg.maps):
            name = name[len(v.prefix) :]  # maps are stored names
            ok = np.isfinite(values[name])
            x = values[name][ok].astype(np.float64)
            for j, w in enumerate((None, x, x**2)):
                sums["maps"][k, j] += np.bincount(cells[ok], weights=w, minlength=n)
        picked.append(pd.DataFrame(values).assign(**{"step": time, cfg.group_by: cell_group}))
    picked = pd.concat(picked, ignore_index=True)
    return _final(GriddedWindows(arrays, picked, counts, sums), cfg)


def _final(out: GriddedWindows, cfg: Gridding) -> GriddedWindows:
    """The outputs under their stored names (variables.prefix). picked keeps the group column
    under its input name too, as stats() groups by it."""
    v = cfg.variables
    if v.prefix:
        out.arrays = {v.final(k): a for k, a in out.arrays.items()}
        if not out.picked.empty:
            names = {k: v.final(k) for k in v.observed()}
            group = out.picked[cfg.group_by]
            out.picked = out.picked.rename(columns=names)
            out.picked[cfg.group_by] = group
    return out
