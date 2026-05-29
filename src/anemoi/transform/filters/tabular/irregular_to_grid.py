# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from datetime import datetime

import earthkit.data as ekd
import numpy as np
import pandas as pd

from anemoi.transform.fields import new_field_from_latitudes_longitudes
from anemoi.transform.fields import new_field_with_valid_datetime
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filter import new_field_from_numpy
from anemoi.transform.filters.tabular import filter_registry
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols
from anemoi.transform.filters.tabular.support.window import Window

LOG = logging.getLogger(__name__)


@filter_registry.register("irregular_to_grid")
class IrregularToGrid(Filter):
    """Convert irregular observations within a time window to an ekd.FieldList of gridded fields.

    For each target time step, observations are selected from a configurable window
    and the observation nearest to the target time is chosen per grid point.

    All columns listed under ``columns`` must be present in the input DataFrame,
    in addition to the columns ``date`` and ``spatial_index``.

    Parameters
    ----------
    template : str
        Path to a template file used to construct output fields.
    start_time : datetime
        Start of the time range.
    end_time : datetime
        End of the time range.
    columns : list[str]
        Column names in the input DataFrame to grid.
    time_freq : str
        Frequency of target time steps (default ``"6h"``).
    grid : str
        Named grid specification (default ``"o96"``).
    window: str or None
        String representation of the time window around each target time, e.g. ``"(-6h, 0h]"``.
        Defaults to ``"(-time_freq, 0]"`` when ``None``.

    Notes
    -----
      - Grid points with no observations in the window are filled with ``NaN``.
      - ``spatial_index`` must be a valid integer index into the named grid.
      - Out-of-range spatial indices are silently ignored.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - irregular_to_grid:
              template: "path/to/template.grib"
              start_time: "2020-01-01"
              end_time: "2020-01-31"
              columns: [t, q, u, v]
              time_freq: "6h"
              grid: "o96"

    """

    def __init__(
        self,
        template: str,
        start_time: datetime,
        end_time: datetime,
        columns: list[str],
        time_freq: str = "6h",
        grid: str = "o96",
        window: str | None = None,
    ):
        self.template = template
        self.start_time = start_time
        self.end_time = end_time
        self.columns = columns
        self.time_freq = time_freq
        self.grid = grid

        if not self.columns:
            raise ValueError("At least one column must be specified")

        window = window or f"(-{time_freq}, 0]"
        self.window = Window(window)

    def forward(self, df: pd.DataFrame) -> ekd.FieldList:
        """Convert irregular values (e.g. observations) within a time window to gridded arrays.

        Parameters:
        -----------
        df : pandas DataFrame
            Must have columns: ['date', 'spatial_index'] and the columns
            specified in self.columns

        Returns:
        --------
        ekd.FieldList
            The gridded fields computed from the DataFrame.
        """
        # Read template field
        template_field = ekd.from_source("file", self.template)

        # Ensure date column is datetime-like
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Check that all requested columns exist
        required_cols = ["date", "spatial_index"] + list(self.columns)
        raise_if_df_missing_cols(df, required_cols=required_cols)

        # Generate grid coordinates
        LOG.info(f"Generating grid coordinates for {self.grid}...")
        grid_lats, grid_lons = self._define_grid(self.grid)
        n_spatial_total = len(grid_lats)
        LOG.info(f"Generated grid with {n_spatial_total} points")

        # Create target times
        target_times = pd.date_range(start=self.start_time, end=self.end_time, freq=self.time_freq)

        # Initialize grids for all columns to grid with NaNs
        # NaNs will remain for time steps with no observations
        n_time = len(target_times)
        grids = {col: np.full((n_time, n_spatial_total), np.nan) for col in self.columns}

        # Process each target time
        for t_idx, target_time in enumerate(target_times):
            # Convert target_time to tz-naive for consistent comparison (all times assumed UTC)
            target_time_naive = pd.Timestamp(target_time).tz_localize(None)

            df_window = self.select_window(df, target_time_naive, self.columns, self.window)
            if df_window is None:
                # No observations - NaNs remain in grids for this time step
                continue

            df_nearest = self.get_nearest_obs(df_window, target_time_naive)
            self._fill_grids(grids, df_nearest, self.columns, n_spatial_total, t_idx)

        return self._build_output_fieldlist(template_field, target_times, grid_lats, grid_lons, grids)

    @staticmethod
    def _build_output_fieldlist(
        template: ekd.Field,
        times: pd.DatetimeIndex,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        grids: dict[str, np.ndarray],
    ) -> ekd.FieldList:
        fields = []
        for t, time in enumerate(times):
            for param, arr in grids.items():
                field = new_field_from_numpy(arr[t], template=template, param=param)
                field = new_field_with_valid_datetime(field, time)
                field = new_field_from_latitudes_longitudes(field, latitudes, longitudes)
                fields.append(field)
        return new_fieldlist_from_list(fields)

    @staticmethod
    def _fill_grids(
        grids: dict[str, np.ndarray],
        df_nearest: pd.DataFrame,
        columns: list[str],
        n_spatial_total: int,
        t_idx: int,
    ):
        # Note: mutates grids in-place
        # Fill in the grids (spatial_index is used directly as array index
        spatial_indices = df_nearest["spatial_index"].values.astype(np.intp)
        valid_mask = (spatial_indices >= 0) & (spatial_indices < n_spatial_total)
        valid_indices = spatial_indices[valid_mask]

        for col in columns:
            col_values = df_nearest[col].values[valid_mask]
            grids[col][t_idx, valid_indices] = col_values

    @staticmethod
    def select_window(
        df: pd.DataFrame,
        target_time: datetime,
        columns: list[str],
        window: Window,
    ) -> pd.DataFrame | None:
        """Select observations within a time window around the target time.

        Parameters
        ----------
        df : pd.DataFrame
            Input observations with a ``"date"`` column.
        target_time : datetime
            Centre of the selection window.
        columns : list[str]
            Data columns — rows where *all* of these are NaN are dropped.
        window: Window
            Window definition used to select observations.

        Returns
        -------
        pd.DataFrame or None
            Filtered observations, or ``None`` if no valid observations remain.
        """
        window_start = target_time + window.before
        window_end = target_time + window.after
        before_closed, after_closed = window.closed
        before_select = df["date"].ge if before_closed else df["date"].gt
        after_select = df["date"].le if after_closed else df["date"].lt

        # Get observations in this window
        mask = before_select(window_start) & after_select(window_end)
        if not mask.any():
            # No observations in this window
            return None

        df_window = df[mask].copy()

        # Filter out rows where ALL columns to grid are NaN
        valid_mask = ~df_window[columns].isna().all(axis=1)
        df_window = df_window[valid_mask]

        if len(df_window) == 0:
            # All observations filtered out
            return None
        return df_window

    @staticmethod
    def get_nearest_obs(df_window: pd.DataFrame, target_time: datetime) -> pd.DataFrame:
        # For each spatial location, select observation nearest to target_time
        df_window["time_diff"] = (df_window["date"] - target_time).abs()

        # Group by spatial_index and take the observation closest to target_time
        idx_nearest = df_window.groupby("spatial_index")["time_diff"].idxmin()
        df_nearest = df_window.loc[idx_nearest]
        return df_nearest

    @staticmethod
    def _define_grid(grid: str) -> tuple[np.ndarray, np.ndarray]:
        from anemoi.transform.grids.named import lookup

        grid_info = lookup(grid)
        lon = np.where(grid_info["longitudes"] > 180, grid_info["longitudes"] - 360, grid_info["longitudes"])
        return grid_info["latitudes"], lon
