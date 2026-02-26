# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pandas as pd
from scipy.spatial import cKDTree

from anemoi.transform.filter import Filter
from anemoi.transform.filters.tabular import filter_registry


@filter_registry.register("assign_to_grid")
class AssignToGrid(Filter):
    """Adds a new column ('grid_index_{grid}') to the DataFrame which represents
    the index of the nearest grid point, based on the latitude/longitude
    coordinates.

    The config item `grid` is a string describing the specification of the grid.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - assign_to_grid:
              grid: o96

    """

    def __init__(self, *, grid: str):
        if not grid:
            raise ValueError("No grid specified.")
        self.grid = grid

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        from anemoi.transform.filters.tabular.support.superob import define_grid
        from anemoi.transform.filters.tabular.support.superob import define_healpix_grid

        # Define output grid based on grid type
        if self.grid[0] == "h":
            # Healpix grid
            nside = int(self.grid[1:])
            grid_points = define_healpix_grid(nside)
        else:
            # Regular grid (N320, O1280, etc.)
            grid_points = define_grid(self.grid)

        # Create KDTree for fast nearest-neighbor search
        tree = cKDTree(grid_points)

        # Find nearest spatial grid points
        distances, spatial_indices = tree.query(obs_df[["latitude", "longitude"]])

        # Add grid_index column (spatial only, no temporal component)
        return obs_df.assign(**{f"grid_index_{self.grid}": spatial_indices}, distance=distances)
