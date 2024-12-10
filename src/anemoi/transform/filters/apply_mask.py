# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd
import numpy as np

from . import filter_registry
from .base import SimpleFilter


def masks(var, mask):
    var[mask] = np.nan
    return var


@filter_registry.register("glacier_mask")
class SnowDepthMasked(SimpleFilter):
    """A filter to mask about glacier in snow depth."""

    def __init__(
        self,
        *,
        glacier_mask,
        snow_depth="sd",
        snow_depth_masked="sd_masked",
    ):
        self.glacier_mask = ekd.from_source("file", glacier_mask)[0].to_numpy().astype(bool)
        self.snow_depth = snow_depth
        self.snow_depth_masked = snow_depth_masked

    def forward(self, data):
        return self._transform(
            data,
            self.forward_transform,
            self.snow_depth,
        )

    def backward(self, data):
        raise NotImplementedError("SnowDepthMasked is not reversible")

    def forward_transform(self, sd):
        """Mask out glaciers in snow depth"""

        snow_depth_masked = masks(sd.to_numpy(), self.glacier_mask)

        yield self.new_field_from_numpy(snow_depth_masked, template=sd, param=self.snow_depth_masked)

    def backward_transform(self, sd):
        raise NotImplementedError("SnowDepthMasked is not reversible")
