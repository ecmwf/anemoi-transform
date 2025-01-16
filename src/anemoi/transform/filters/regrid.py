# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections import defaultdict

from earthkit.data.core.fieldlist import Field

from ..fields import new_field_from_numpy
from ..fields import new_fieldlist_from_list
from ..filter import Filter
from . import filter_registry

LOG = logging.getLogger(__name__)


def as_gridspec(grid):
    if grid is None:
        return None
    if isinstance(grid, str):
        return {"grid": grid}
    return grid


def as_griddata(grid):
    if isinstance(grid, Field):
        lat, lon = grid.gridpoints()
        return dict(latitudes=lat, longitudes=lon)
    if isinstance(grid, dict) and "latitudes" in grid and "longitudes" in grid:
        return grid
    if isinstance(grid, str):
        from anemoi.utils.grids import grids

        return grids(grid)
    raise ValueError(f"Invalid grid: {grid}")


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
        self.in_grid = as_gridspec(in_grid)
        self.out_grid = as_gridspec(out_grid)
        self.method = method

    def __call__(self, field):
        from earthkit.regrid import interpolate

        return new_field_from_numpy(
            interpolate(
                field.to_numpy(flatten=True),
                in_grid=self.in_grid,
                out_grid=self.out_grid,
                method=self.method,
            ),
            template=field,
        )


class AnemoiInterpolator:
    """Interpolator tools for the grids that are not supported yet by earthkit."""

    nearest_grid_points = None

    def __init__(self, in_grid, out_grid, method):
        if method != "nearest":
            raise NotImplementedError(f"AnemoiInterpolator does not support {method}, only 'nearest'")

        self.ingrid = as_griddata(in_grid)
        self.outgrid = as_griddata(out_grid)

        if self.outgrid is None:
            raise ValueError("out_grid is required, but not provided")

    def __call__(self, field):
        if self.ingrid is None:
            self.ingrid = as_griddata(field)
            assert self.ingrid is not None, field

        if self.nearest_grid_points is None:
            from anemoi.utils.grids import nearest_grid_points

            self.nearest_grid_points = nearest_grid_points(
                self.ingrid["latitudes"],
                self.ingrid["longitudes"],
                self.outgrid["latitudes"],
                self.outgrid["longitudes"],
            )

        data = field.to_numpy(flatten=True)
        assert data.shape == self.ingrid["latitudes"].shape, (data.shape, self.ingrid["latitudes"].shape)
        assert data.shape == self.ingrid["longitudes"].shape, (data.shape, self.ingrid["longitudes"].shape)

        data = data[..., self.nearest_grid_points]
        return new_field_from_numpy(data, template=field)


def interpolator(in_grid, out_grid, method):
    INTERPOLATORS = defaultdict(lambda: EarthkitInterpolator)
    INTERPOLATORS["nearest"] = AnemoiInterpolator
    return INTERPOLATORS[method](in_grid, out_grid, method)
