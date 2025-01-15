# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from ..fields import new_field_from_numpy
from ..fields import new_fieldlist_from_list
from ..filter import Filter
from . import filter_registry

LOG = logging.getLogger(__name__)


@filter_registry.register("regrid")
class RegridFilter(Filter):
    """A filter to regrid fields using earthkit-regrid."""

    def __init__(self, *, in_grid=None, out_grid=None, method="linear"):
        self.in_grid = in_grid
        self.out_grid = out_grid
        self.method = method
        self.interpolator = interpolator(in_grid, out_grid, method)

    def forward(self, data):
        return self._interpolate(data, self.in_grid, self.out_grid, self.method)

    def backward(self, data):
        return self._interpolate(data, self.out_grid, self.in_grid, self.method)

    def _interpolate(self, data, in_grid, out_grid, method):

        result = []
        for field in data:
            result.append(self.interpolator(field))

        return new_fieldlist_from_list(result)


class EarthkitInterpolator:
    """Default interpolator using earthkit."""

    def __init__(self, in_grid, out_grid, method):
        self.in_grid = in_grid
        self.out_grid = out_grid
        self.method = method

    def __call__(self, field):
        from earthkit.regrid import interpolate

        return new_field_from_numpy(
            interpolate(
                field.to_numpy(flatten=True),
                in_grid={"grid": self.in_grid},
                out_grid={"grid": self.out_grid},
                method=self.method,
            ),
            template=field,
        )


class AnemoiInterpolator:
    """Interpolator tools for the grids that are not supported yet by earthkit."""

    def __init__(self, in_grid, out_grid, method):
        if method != "nearest":
            raise NotImplementedError(f"AnemoiInterpolator does not support {method}, only 'nearest'")

        from anemoi.utils.grids import grids
        from anemoi.utils.grids import nearest_grid_points

        ingrid = grids(in_grid)
        outgrid = grids(out_grid)

        self.in_shape = ingrid["latitudes"].shape  # for checking the shape of the input data

        self._nearest_grid_points = nearest_grid_points(
            ingrid["latitudes"],
            ingrid["longitudes"],
            outgrid["latitudes"],
            outgrid["longitudes"],
        )

    def __call__(self, field):
        data = field.to_numpy(flatten=True)
        assert data.shape == self.in_shape, (data.shape, self.in_shape)
        data = data[..., self._nearest_grid_points]
        return new_field_from_numpy(data, template=field)


def interpolator(in_grid, out_grid, method):
    if method.startswith("anemoi."):
        method = method.split(".")[1]
        return AnemoiInterpolator(in_grid, out_grid, method)
    return EarthkitInterpolator(in_grid, out_grid, method)
