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

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching

from .rodeo_opera_preprocessing import clip_opera

FACTOR_TP = 1000  # convert from mm to m

MAX_TP = 10000  # clip TP


@filter_registry.register("rodeo_opera_clipping")
class RodeoOperaClipping(MatchingFieldsFilter):
    """A filter to clip reprojected data in Rodeo Opera data.

    Parameters
    ----------
    total_precipitation : str, optional
        The name of the total_precipitation field, by default "tp".
    max_total_precipitation : int, optional
        The maximum value for tp, by default MAX_TP.
    quality : ekd.Field
        The quality data.
    mask : ekd.Field
        The mask data.
    """

    @matching(
        select="param",
        forward=("total_precipitation", "quality", "mask"),
    )
    def __init__(
        self,
        *,
        total_precipitation: str = "tp",
        max_total_precipitation: int = MAX_TP,
        quality: str = "qi",
        mask: str = "dm",
    ) -> None:
        """Initialize the RodeoOperaPreProcessing filter.

        Parameters
        ----------
        tp : str, optional
            The name of the tp field, by default "tp".
        max_total_precipitation : int, optional
            The maximum value for tp, by default MAX_TP.
        quality : ekd.Field
            The quality data.
        mask : ekd.Field
            The mask data.
        """
        self.total_precipitation = total_precipitation
        self.max_total_precipitation = max_total_precipitation
        self.quality = quality
        self.mask = mask

    def forward_transform(
        self,
        total_precipitation: ekd.Field,
        quality,
        mask,
    ) -> Iterator[ekd.Field]:
        """Pre-process Rodeo Opera data.

                Parameters
                ----------
                total_precipitation : ekd.Field
                    The tp data.
                quality : ekd.Field
                    The quality data.
                mask : ekd.Field
                    The mask data.
                Returns
                -------
                Iterator[ekd.Field]
                    Transformed fields.
        |
        """
        total_precipitation_cleaned, quality_clipped = clip_opera(
            tp=total_precipitation.to_numpy(),
            quality=quality.to_numpy(),
            max_total_precipitation=self.max_total_precipitation,
        )

        total_precipitation_cleaned = total_precipitation_cleaned / FACTOR_TP

        yield self.new_field_from_numpy(
            total_precipitation_cleaned, template=total_precipitation, param=self.total_precipitation
        )
        yield self.new_field_from_numpy(quality_clipped, template=quality, param=self.quality)
        yield mask
