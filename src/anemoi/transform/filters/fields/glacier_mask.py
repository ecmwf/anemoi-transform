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

from anemoi.transform.filter import SingleFieldFilter
from anemoi.transform.filters import filter_registry


def mask_glaciers(snow_depth: np.ndarray, glacier_mask: np.ndarray) -> np.ndarray:
    """Mask out glaciers in snow depth data.

    Parameters
    ----------
    snow_depth : np.ndarray
        Array of snow depth values.
    glacier_mask : np.ndarray
        Boolean array indicating glacier locations.

    Returns
    -------
    np.ndarray
        Snow depth array with glaciers masked out.
    """
    snow_depth[glacier_mask] = np.nan
    return snow_depth


@filter_registry.register("glacier_mask")
class SnowDepthMasked(SingleFieldFilter):
    """A filter to mask about glacier in snow depth."""

    required_inputs = ("glacier_mask",)
    optional_inputs = {"snow_depth": "sd", "snow_depth_masked": "sd_masked"}

    def prepare_filter(self):
        self.glacier_mask = ekd.from_source("file", self.glacier_mask)[0].to_numpy().astype(bool)

    def forward_select(self):
        return {"param": self.snow_depth}

    def forward_transform(self, snow_depth: ekd.Field) -> ekd.Field:
        """Mask out glaciers in snow depth.

        Parameters
        ----------
        snow_depth : ekd.Field
            Snow depth field.

        Returns
        -------
        ekd.Field
            Snow depth field with glaciers masked out.
        """
        snow_depth_masked = mask_glaciers(snow_depth.to_numpy(), self.glacier_mask)

        return self.new_field_from_numpy(snow_depth_masked, template=snow_depth, param=self.snow_depth_masked)
