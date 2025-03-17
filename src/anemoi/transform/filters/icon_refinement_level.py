# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

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

    def __init__(self, *, grid, refinement_level_c):
        self.grid = grid
        self.refinement_level_c = refinement_level_c

        self.latitudes, self.longitudes = icon_grid(self.grid, self.refinement_level_c)
        self.nearest_grid_points = None

    def forward(self, fields):
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
            result.append(
                new_field_from_latitudes_longitudes(
                    new_field_from_numpy(data, template=field), latitudes=self.latitudes, longitudes=self.longitudes
                )
            )

        return new_fieldlist_from_list(result)

    def backward(self, data):
        raise NotImplementedError("IconRefinement is not reversible")
