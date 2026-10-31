# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Rows of several inputs → the one frame gridding reads: each input's own dates, groups
(satellites), channel slots, group offset and scan geometry. Pure, as gridding.

A dataset can combine instruments (MSU before AMSU-A), products (a CDR and the real-time
stream) or one stream's instrument versions (HIRS/2 bands in their own slots). Each input
fills the dataset channels it names; the others are missing in its rows.
"""

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic import model_validator

from anemoi.transform.obs import derived
from anemoi.transform.obs.config import Gridding
from anemoi.transform.obs.config import Scan
from anemoi.transform.obs.config import Strict


class Input(Strict):
    """How one input's rows join the dataset.

    dates: [start, end) of the rows kept (either open if unset); an input outside the
    dates of a read is not read. groups: keep rows whose group column (before the offset)
    is listed, all if empty. channels: the dataset channels this input fills: every one
    from its same-named column (unset), those listed, or {input column: dataset channel}.
    group_offset: added to the group column, e.g. 10000 to tell a CDR's satellites from the
    real-time stream's. scan: this input's scan geometry, when the dataset's is "input".
    """

    dates: tuple[str | None, str | None] = (None, None)
    groups: list[int] = Field(default_factory=list)
    channels: list[str] | dict[str, str] | None = None
    group_offset: int = 0
    scan: Scan | None = None

    @model_validator(mode="after")
    def _check(self):
        if isinstance(self.channels, dict) and len(set(self.channels.values())) != len(self.channels):
            raise ValueError("input channels: two columns fill the same dataset channel")
        start, end = (pd.Timestamp(d) if d else None for d in self.dates)
        if start is not None and end is not None and start >= end:
            raise ValueError(f"input dates: start {start} is not before end {end}")
        return self

    def overlaps(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """Whether rows from start to end can be kept."""
        lo, hi = (pd.Timestamp(d) if d else None for d in self.dates)
        return (hi is None or pd.Timestamp(start) < hi) and (lo is None or pd.Timestamp(end) >= lo)

    def source_of(self, channel: str) -> str | None:
        """The input column that fills a dataset channel, or None."""
        if self.channels is None:
            return channel
        if isinstance(self.channels, list):
            return channel if channel in self.channels else None
        return {v: k for k, v in self.channels.items()}.get(channel)

    def check(self, cfg: Gridding) -> None:
        channels = cfg.variables.channels
        named = self.channels.values() if isinstance(self.channels, dict) else self.channels or []
        if unknown := set(named) - set(channels):
            raise ValueError(f"input channels names no dataset channel: {sorted(unknown)}")
        if cfg.variables.scan == "input" and self.scan is None:
            raise ValueError('variables.scan is "input": every input needs its scan')
        if self.scan is not None and cfg.variables.scan != "input":
            raise ValueError('an input scan needs variables.scan: "input"')

    def columns(self, cfg: Gridding) -> list[str]:
        """The input columns to read for cfg.input_columns()."""
        self.check(cfg)
        channels = set(cfg.variables.channels)
        out = []
        for column in cfg.input_columns():
            if column in channels:
                column = self.source_of(column)
            elif column == "scan_pos" and cfg.variables.scan == "input":
                column = self.scan.column
            if column is not None:
                out.append(column)
        return list(dict.fromkeys(out))

    def prepare(self, cfg: Gridding, df: pd.DataFrame) -> pd.DataFrame:
        """The input's rows kept, with exactly cfg.input_columns() as columns."""
        self.check(cfg)
        keep = np.ones(len(df), bool)
        start, end = (pd.Timestamp(d) if d else None for d in self.dates)
        if start is not None:
            keep &= (df["date"] >= start).to_numpy()
        if end is not None:
            keep &= (df["date"] < end).to_numpy()
        if self.groups:
            keep &= df[cfg.group_by].isin(self.groups).to_numpy()
        df = df[keep].reset_index(drop=True)
        out = {}
        for column in cfg.input_columns():
            if column in cfg.variables.channels:
                source = self.source_of(column)
                out[column] = df[source] if source else np.full(len(df), np.nan, np.float32)
            elif column == "scan_pos" and cfg.variables.scan == "input":
                out[column] = derived.scan_position(df[self.scan.column].to_numpy(np.int64), self.scan)
            elif column == cfg.group_by:
                out[column] = df[column].astype(np.int64) + self.group_offset
            else:
                out[column] = df[column]  # keeps its dtype (pandas nullable integers)
        return pd.DataFrame(out)


def combine(cfg: Gridding, parts: list[tuple[Input, pd.DataFrame]]) -> pd.DataFrame:
    """One frame of every input's prepared rows (Input.prepare)."""
    frames = [inp.prepare(cfg, df) for inp, df in parts]
    if not frames:
        return pd.DataFrame(
            {c: pd.Series(dtype="datetime64[ns]" if c == "date" else float) for c in cfg.input_columns()}
        )
    return pd.concat(frames, ignore_index=True)
