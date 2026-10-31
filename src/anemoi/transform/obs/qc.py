# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""QC before gridding: which rows may be selected, which values are masked.

Rows are kept (with a mask) rather than dropped, so the counts and per-cell rejection maps
see every row.
"""

import numpy as np
import pandas as pd

from anemoi.transform.obs.config import Qc


def apply(
    qc: Qc, df: pd.DataFrame, channels: list[str], group_by: str = "satid"
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """(rows with masked values, keep mask, counts).

    counts: one row per (group, step) with n = rows (flag:<col>, bits:<col>, incomplete, in,
    kept) or values (value_flag:<channel>, range:<col>, known_bad:<i>) removed by that step.
    A row failing several flags counts under the first.
    """
    keep = np.ones(len(df), bool)
    group = df[group_by].to_numpy(dtype=np.int64)
    steps: list[tuple[str, np.ndarray]] = [("in", keep.copy())]
    for column, values in qc.flags.items():
        ok = df[column].isin(values).to_numpy()
        steps.append((f"flag:{column}", keep & ~ok))
        keep &= ok
    for column, mask in qc.bits.items():
        ok = (df[column].to_numpy(dtype=np.int64) & mask) == 0
        steps.append((f"bits:{column}", keep & ~ok))
        keep &= ok
    masked = {}
    for channel, column in qc.value_flags.items():
        v = df[channel].to_numpy(dtype=np.float32, na_value=np.nan)
        out = np.isfinite(v) & (df[column].to_numpy(dtype=np.int64) != 0)
        steps.append((f"value_flag:{channel}", keep & out))
        masked[channel] = np.where(out, np.nan, v)
    if masked:
        df = df.assign(**masked)
        masked = {}
    for column, (lo, hi) in qc.range.items():
        v = df[column].to_numpy(dtype=np.float32, na_value=np.nan)
        out = (v < lo) | (v > hi)
        steps.append((f"range:{column}", keep & out))
        masked[column] = np.where(out, np.nan, v)
    if masked:
        df = df.assign(**masked)
    for i, bad in enumerate(qc.known_bad):
        rows = np.ones(len(df), bool)
        if bad.groups:
            rows &= np.isin(group, bad.groups)
        if bad.start:
            rows &= (df["date"] >= pd.Timestamp(bad.start)).to_numpy()
        if bad.end:
            rows &= (df["date"] < pd.Timestamp(bad.end)).to_numpy()
        if rows.any():
            df = df.assign(**{c: df[c].where(~rows) for c in bad.columns})
        n_values = np.zeros(len(df), np.int64)
        n_values[keep & rows] = len(bad.columns)
        steps.append((f"known_bad:{i}", n_values))
    required = qc.required(channels)
    if required != []:
        present = np.isfinite(df[required or channels].to_numpy(dtype=np.float32, na_value=np.nan))
        complete = present.any(1) if required is None else present.all(1)
        steps.append(("incomplete", keep & ~complete))
        keep &= complete
    steps.append(("kept", keep))
    groups, index = np.unique(group, return_inverse=True)
    rows = [
        {"group": int(g), "step": name, "n": int(n)}
        for name, x in steps
        for g, n in zip(groups, np.bincount(index, weights=x, minlength=groups.size))
    ]
    return df, keep, pd.DataFrame(rows, columns=["group", "step", "n"])
