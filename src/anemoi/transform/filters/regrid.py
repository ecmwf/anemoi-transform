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

    def forward(self, data):
        return self._interpolate(data, self.in_grid, self.out_grid, self.method)

    def backward(self, data):
        return self._interpolate(data, self.out_grid, self.in_grid, self.method)

    def _interpolate(self, data, in_grid, out_grid, method):
        from earthkit.regrid import interpolate

        result = []
        for field in data:
            result.append(
                new_field_from_numpy(
                    interpolate(
                        field.to_numpy(flatten=True),
                        in_grid={"grid": in_grid},
                        out_grid={"grid": out_grid},
                        method=method,
                    ),
                    template=field,
                )
            )

        return new_fieldlist_from_list(result)
