# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pandas as pd

from anemoi.transform.filter import Filter
from anemoi.transform.filters.tabular import filter_registry
from anemoi.transform.filters.tabular.support.superob import assign_nearest_grid
from anemoi.transform.filters.tabular.support.superob import define_grid
from anemoi.transform.filters.tabular.support.superob import define_healpix_grid


@filter_registry.register("superob")
class SuperOb(Filter):
    """SuperOb filter for aggregating observations into grid cells.

    The configuration should be a dictionary with the following keys:
    - grid: str, the grid to use for aggregation
    - timeslot_length: int, the length of the timeslot in seconds
    - columns_to_take_nearest: list[str], the columns to take the nearest value for
    - columns_to_groupby: list[str], the columns to group by for aggregation

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - superob:
              grid: o06
              timeslot_length: 3600
              columns_to_take_nearest: ["datetime"]
              columns_to_groupby: ["reporttype"]

    """

    def __init__(
        self,
        *,
        grid: str,
        timeslot_length: int,
        columns_to_take_nearest: list[str] | None = None,
        columns_to_groupby: list[str] | None = None,
    ):
        self.grid = grid
        self.timeslot_length = timeslot_length
        self.columns_to_take_nearest = columns_to_take_nearest if columns_to_take_nearest else []
        self.columns_to_groupby = columns_to_groupby if columns_to_groupby else []

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.grid == "native" or len(df) == 0:
            return df

        # Define output grid (spatail + temporal)
        if self.grid[0] == "h":
            output_grid = define_healpix_grid(int(self.grid[1:]))
        else:
            output_grid = define_grid(self.grid)

        df.dropna(subset=["datetime", "latitude", "longitude"], inplace=True)
        if len(df) == 0:
            return df

        # Assign each observation to an output grid cell
        df = assign_nearest_grid(df, output_grid, self.timeslot_length)

        # build groupby cols once
        groupby_cols = ["grid_index"]
        if self.columns_to_groupby:
            groupby_cols.extend(self.columns_to_groupby)

        # Average only non-key, non-nearest columns
        columns_to_average = [c for c in df.columns if c not in (set(groupby_cols) | set(self.columns_to_take_nearest))]

        # Keep keys as index on both frames; align by index (no big merge keys)
        averaged_df = df.groupby(groupby_cols, observed=True, sort=False)[columns_to_average].mean()

        nearest_idx = df.groupby(groupby_cols, observed=True, sort=False)["distance"].idxmin()
        nearest_df = df.loc[nearest_idx, self.columns_to_take_nearest + groupby_cols].set_index(groupby_cols)

        # Drop duplicates from the index before concatenation
        averaged_df = averaged_df[~averaged_df.index.duplicated(keep="first")]
        nearest_df = nearest_df[~nearest_df.index.duplicated(keep="first")]
        gridded_df = pd.concat([averaged_df, nearest_df], axis=1, join="inner").reset_index()
        gridded_df = gridded_df.drop(columns=["grid_index", "distance"], errors="ignore")
        gridded_df = gridded_df.sort_values("datetime")
        return gridded_df
