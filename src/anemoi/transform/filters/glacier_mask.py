# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Iterator

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching


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
class SnowDepthMasked(MatchingFieldsFilter):
    """A filter to mask about glacier in snow depth."""

    @matching(select="param", forward="snow_depth")
    def __init__(
        self,
        *,
        glacier_mask: str,
        snow_depth: str = "sd",
        snow_depth_masked: str = "sd_masked",
    ) -> None:
        """Initialize the SnowDepthMasked filter.

        Parameters
        ----------
        glacier_mask : str
            Path to the glacier mask file.
        snow_depth : str, optional
            Name of the snow depth parameter, by default "sd".
        snow_depth_masked : str, optional
            Name of the masked snow depth parameter, by default "sd_masked".
        """

        self.snow_depth = snow_depth
        self.glacier_mask = ekd.from_source("file", glacier_mask)[0].to_numpy().astype(bool)
        self.snow_depth_masked = snow_depth_masked

    def forward_transform(self, snow_depth: ekd.Field) -> Iterator[ekd.Field]:
        """Mask out glaciers in snow depth.

        Parameters
        ----------
        snow_depth : ekd.Field
            Snow depth field.

        Returns
        -------
        Iterator[ekd.Field]
            Snow depth field with glaciers masked out.
        """
        snow_depth_masked = mask_glaciers(snow_depth.to_numpy(), self.glacier_mask)

        yield self.new_field_from_numpy(snow_depth_masked, template=snow_depth, param=self.snow_depth_masked)
