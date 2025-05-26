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

NODATA = -9.999e06
UNDETECTED = -8.888e06

_NODATA = 1
_UNDETECTED = 2
_INF = 3

MAX_TP = 10000
MAX_QI = 1


def clip_opera(
    tp: np.ndarray, quality: np.ndarray = None, max_total_precipitation: int = MAX_TP
) -> tuple[np.ndarray, np.ndarray]:
    """Clip the tp and quality arrays to specified maximum values.

    Parameters
    ----------
    tp : numpy.ndarray
        The tp array to be clipped.
    quality : numpy.ndarray
        The quality array to be clipped.
    max_total_precipitation : int
        The maximum value for tp.

    Returns
    -------
    tuple
        A tuple containing the clipped tp and quality arrays.
    """
    tp[tp < 0] = 0
    tp[tp >= max_total_precipitation] = max_total_precipitation
    if quality is not None:
        quality[quality >= MAX_QI] = MAX_QI
        return tp, quality
    return tp


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


@filter_registry.register("rodeo_opera_preprocessing")
class RodeoOperaPreProcessing(MatchingFieldsFilter):
    """A filter to select only good quality data in Rodeo Opera data.

    Parameters
    ----------
    total_precipitation : str, optional
        The name of the total_precipitation field, by default "tp".
    quality : str, optional
        The name of the quality field, by default "quality".
    mask : str, optional
        The name of the mask field, by default "mask".
    max_total_precipitation : int, optional
        The maximum value for tp, by default MAX_TP.
    """

    @matching(
        select="param",
        forward=("total_precipitation", "quality", "mask"),
    )
    def __init__(
        self,
        *,
        total_precipitation: str = "tp",
        quality: str = "qi",
        mask: str = "dm",
        max_total_precipitation: int = MAX_TP,
    ) -> None:
        """Initialize the RodeoOperaPreProcessing filter.

        Parameters
        ----------
        tp : str, optional
            The name of the tp field, by default "tp".
        quality : str, optional
            The name of the quality field, by default "quality".
        mask : str, optional
            The name of the mask field, by default "mask".
        max_total_precipitation : int, optional
            The maximum value for tp, by default MAX_TP.
        """
        self.total_precipitation = total_precipitation
        self.quality = quality
        self.mask = mask
        self.max_total_precipitation = max_total_precipitation

    def forward_transform(
        self,
        total_precipitation: ekd.Field,
        quality: ekd.Field,
        mask: ekd.Field,
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
        """
        # 1st - apply masking
        total_precipitation_masked = mask_opera(
            tp=total_precipitation.to_numpy(), quality=quality.to_numpy(), mask=mask.to_numpy()
        )

        # 2nd - apply clipping
        total_precipitation_cleaned, quality_clipped = clip_opera(
            tp=total_precipitation_masked,
            quality=quality.to_numpy(),
            max_total_precipitation=self.max_total_precipitation,
        )

        yield self.new_field_from_numpy(
            total_precipitation_cleaned, template=total_precipitation, param=self.total_precipitation
        )
        yield self.new_field_from_numpy(quality_clipped, template=quality, param=self.quality)
        yield mask
