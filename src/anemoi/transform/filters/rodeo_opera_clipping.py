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
from anemoi.transform.filters.rodeo_opera_clipping import clip_opera

MAX_TP = 10000


def mask_opera(tp: np.ndarray, quality: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply masking to the tp array based on the mask array.

    Parameters
    ----------
    tp : numpy.ndarray
        The tp array to be masked.
    quality : numpy.ndarray
        The quality array.
    mask : numpy.ndarray
        The mask array indicating which values to mask.

    Returns
    -------
    numpy.ndarray
        The masked tp array.
    """
    print("✅✅", quality)
    print("✅✅✅", tp)

    # # RAW HDF5 DATA FILTERING
    # tp[quality == NODATA] = np.nan
    # tp[quality == UNDETECTED] = 0

    tp[mask == _NODATA] = np.nan
    tp[mask == _UNDETECTED] = 0
    tp[mask == _INF] = np.nan

    return tp


@filter_registry.register("rodeo_opera_clipping")
class RodeoOperaClipping(MatchingFieldsFilter):
    """A filter to clip reprojected data in Rodeo Opera data.

    Parameters
    ----------
    total_precipitation : str, optional
        The name of the total_precipitation field, by default "tp".
    max_total_precipitation : int, optional
        The maximum value for tp, by default MAX_TP.
    """

    @matching(
        match="param",
        forward=("total_precipitation"),
    )
    def __init__(
        self,
        *,
        total_precipitation: str = "tp",
        max_total_precipitation: int = MAX_TP,
    ) -> None:
        """Initialize the RodeoOperaPreProcessing filter.

        Parameters
        ----------
        tp : str, optional
            The name of the tp field, by default "tp".
        max_total_precipitation : int, optional
            The maximum value for tp, by default MAX_TP.
        """
        self.total_precipitation = total_precipitation
        self.max_total_precipitation = max_total_precipitation

    def forward_transform(
        self,
        total_precipitation: ekd.Field,
    ) -> Iterator[ekd.Field]:
        """Pre-process Rodeo Opera data.

                Parameters
                ----------
                total_precipitation : ekd.Field
                    The tp data.

                Returns
                -------
                Iterator[ekd.Field]
                    Transformed fields.
        |
        """
        total_precipitation_cleaned = clip_opera(
            tp=total_precipitation.to_numpy(), max_total_precipitation=self.max_total_precipitation
        )

        yield self.new_field_from_numpy(
            total_precipitation_cleaned, template=total_precipitation, param=self.total_precipitation
        )
