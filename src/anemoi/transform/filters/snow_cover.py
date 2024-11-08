# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from . import filter_registry
from .base import SimpleFilter


@filter_registry.register("snow_cover")
class SnowCover(SimpleFilter):
    """A filter to compute snow cover from snow density and snow depth."""

    def __init__(
        self,
        *,
        snow_depth="sd",
        snow_density="rsn",
        snow_cover="snowc",
    ):
        self.snow_depth = snow_depth
        self.snow_density = snow_density
        self.snow_cover = snow_cover

    def forward(self, data):
        return self._transform(
            data,
            self.forward_transform,
            self.snow_depth,
            self.snow_density,
        )

    def backward(self, data):
        raise NotImplementedError("SnowCover is not reversible")

    def forward_transform(self, sd, rsn):
        """Convert snow depth and snow density to snow cover"""

        snow_cover = sd.to_numpy() * rsn.to_numpy()

        yield self.new_field_from_numpy(snow_cover, template=sd, param=self.snow_cover)

    def backward_transform(self, sd, rsn):
        raise NotImplementedError("SnowCover is not reversible")
