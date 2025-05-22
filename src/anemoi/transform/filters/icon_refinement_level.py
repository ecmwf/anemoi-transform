# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

import earthkit.data as ekd
import tqdm

from anemoi.transform.fields import new_field_from_latitudes_longitudes
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry
from anemoi.transform.grids.icon import icon_grid

LOG = logging.getLogger(__name__)


@filter_registry.register("icon_refinement_level")
class IconRefinement(Filter):
    """A filter interpolate its input to an ICON grid."""

    def __init__(self, *, grid: Any, refinement_level_c: Any) -> None:
        """Initialize the IconRefinement filter.

        Parameters
        ----------
        grid : Any
            The grid to use for interpolation.
        refinement_level_c : Any
            The refinement level for the grid.
        """

        self.grid = grid
        self.refinement_level_c = refinement_level_c

        self.latitudes, self.longitudes = icon_grid(self.grid, self.refinement_level_c)
        self.nearest_grid_points = None

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:
        """Interpolate the input fields to an ICON grid.

        Parameters
        ----------
        fields : ekd.FieldList
            List of fields to be interpolated.

        Returns
        -------
        ekd.FieldList
            List of interpolated fields.
        """
        if self.nearest_grid_points is None:
            from anemoi.utils.grids import nearest_grid_points

            # We assume all fields have the same grid
            latitudes, longitudes = fields[0].grid_points()
            self.nearest_grid_points = nearest_grid_points(
                latitudes,
                longitudes,
                self.latitudes,
                self.longitudes,
            )

        result = []
        for field in tqdm.tqdm(fields, desc="Regridding"):

            data = field.to_numpy(flatten=True)
            data = data[..., self.nearest_grid_points]
            new_field = new_field_from_latitudes_longitudes(
                new_field_from_numpy(data, template=field), latitudes=self.latitudes, longitudes=self.longitudes
            )
            new_field.resolution = f"mrl{self.refinement_level_c}"
            result.append(new_field)

        return new_fieldlist_from_list(result)
