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


def clip_opera(tp: np.ndarray, quality: np.ndarray, max_tp: int) -> tuple[np.ndarray, np.ndarray]:
    """Clip the tp and quality arrays to specified maximum values.

    Parameters
    ----------
    tp : numpy.ndarray
        The tp array to be clipped.
    quality : numpy.ndarray
        The quality array to be clipped.
    max_tp : int
        The maximum value for tp.

    Returns
    -------
    tuple
        A tuple containing the clipped tp and quality arrays.
    """
    tp[tp < 0] = 0
    tp[tp >= max_tp] = max_tp
    quality[quality >= MAX_QI] = MAX_QI

    return tp, quality


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
    # tp[quality == UNDETECTED] = np.nan

    tp[mask == _NODATA] = np.nan
    tp[mask == _UNDETECTED] = np.nan
    tp[mask == _INF] = np.nan

    return tp


@filter_registry.register("rodeo_opera_preprocessing")
class RodeoOperaPreProcessing(MatchingFieldsFilter):
    """A filter to select only good quality data in Rodeo Opera data.

    Parameters
    ----------
    tp : str, optional
        The name of the tp field, by default "tp".
    quality : str, optional
        The name of the quality field, by default "quality".
    mask : str, optional
        The name of the mask field, by default "mask".
    output : str, optional
        The name of the output field, by default "tp_cleaned".
    max_tp : int, optional
        The maximum value for tp, by default MAX_TP.
    """

    @matching(
        match="param",
        forward=("tp", "quality", "mask"),
    )
    def __init__(
        self,
        *,
        tp: str = "tp",
        quality: str = "quality",
        mask: str = "mask",
        output: str = "tp_cleaned",
        max_tp: int = MAX_TP,
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
        output : str, optional
            The name of the output field, by default "tp_cleaned".
        max_tp : int, optional
            The maximum value for tp, by default MAX_TP.
        """
        self.tp = tp
        self.quality = quality
        self.tp_cleaned = output
        self.mask = mask
        self.max_tp = max_tp

    def forward_transform(
        self,
        tp: ekd.Field,
        quality: ekd.Field,
        mask: ekd.Field,
    ) -> Iterator[ekd.Field]:
        """Pre-process Rodeo Opera data.

        Parameters
        ----------
        tp : ekd.Field
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
        tp_masked = mask_opera(tp=tp.to_numpy(), quality=quality.to_numpy(), mask=mask.to_numpy())

        # 2nd - apply clipping
        tp_cleaned, quality = clip_opera(tp=tp_masked, quality=quality.to_numpy(), max_tp=self.max_tp)

        yield self.new_field_from_numpy(tp_cleaned, template=tp, param=self.tp_cleaned)
