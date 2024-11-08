# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import cfgrib
import numpy as np

from . import filter_registry
from .base import SimpleFilter

glacier_mask_file = "/home/rdx/data/climate/climate.v021/95_4/cicecap"


def mask_glaciers(snow_depth, glacier_mask_file):
    ds = cfgrib.open_dataset(glacier_mask_file, backend_kwargs={"read_keys": [], "indexpath": ""})
    mask = ds.si10.values
    snow_depth[mask] = np.nan
    return snow_depth


@filter_registry.register("snow_depth_masked")
class SnowDepthMasked(SimpleFilter):
    """A filter to mask about glacier in snow depth."""

    def __init__(
        self,
        *,
        snow_depth="sd",
        snow_depth_masked="sd_masked",
    ):
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

        snow_cover = mask_glaciers(sd.to_numpy())

        yield self.new_field_from_numpy(snow_cover, template=sd, param=self.snow_cover)

    def backward_transform(self, sd, rsn):
        raise NotImplementedError("SnowDepthMasked is not reversible")
