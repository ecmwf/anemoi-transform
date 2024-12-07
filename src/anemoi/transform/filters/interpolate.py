# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from ..filter import Filter
from . import filter_registry

LOG = logging.getLogger(__name__)


def _as_gridspec(grid):
    # Tranform 0.25 to (0.25, 0.25)
    try:
        x = float(grid)
        return (x, x)
    except ValueError:
        pass

    return x


@filter_registry.register("interpolate")
class Interpolate(Filter):
    """A filter to interpolate a field to a new grid, and back."""

    def __init__(self, *, target, source=None, method="linear", engine="earthkit-regrid"):
        self.source = _as_gridspec(source)
        self.target = _as_gridspec(target)
        self.method = method
        self.engine = engine

        if engine != "earthkit-regrid":
            raise ValueError(f"Unknown engine {self.engine}")

    def forward(self, data):
        from earthkit.regrid import interpolate

        return interpolate(data, self.source, self.target, method=self.method)

    def backward(self, data):
        from earthkit.regrid import interpolate

        return interpolate(data, self.target, self.source, method=self.method)
