# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections.abc import Iterator

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching

LOG = logging.getLogger(__name__)

NODATA = -9.999e06
UNDETECTED = -8.888e06

_NODATA = 1
_UNDETECTED = 2
_INF = 3

MAX_TP = 10000
MAX_QI = 1


def _clip_variable(variable: np.ndarray, max_value: float) -> np.ndarray:
    variable[variable < 0] = 0
    variable[variable >= max_value] = max_value
    return variable


def clip_opera(
    tp: np.ndarray, quality: np.ndarray, max_total_precipitation: int = MAX_TP
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
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
    tp = _clip_variable(tp, max_total_precipitation)
    quality = _clip_variable(quality, MAX_QI)
    return tp, quality


def mask_opera(tp: np.ndarray, quality: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    quality[mask == _UNDETECTED] = 0

    if not np.isnan(tp).sum() == np.isnan(quality).sum():
        msg = f"Mismatch between NaNs on tp {np.isnan(tp).sum()} and qi {np.isnan(quality).sum()}"
        LOG.warning(msg)

    return tp, quality


@filter_registry.register("rodeo_opera_preprocessing")
class RodeoOperaPreProcessing(MatchingFieldsFilter):
    """A filter to select only good quality data in Rodeo Opera data.

    The ``rodeo_opera_preprocessing`` function applies filtering to the
    OPERA Pan-European composites. This preprocessing consists of:

    -  Masking of undetected pixels using the ``mask`` variable

    -  Clipping of precipitation values to the range ``[0,
       max_total_precipitation]``, where ``max_total_precipitation`` is
       defined at the configuration level. If no value is passed a default
       value (``MAX_TP``) of 10000 is used.

    -  Clipping of the quality index to the range ``[0, 1]``

       By default the ``mask`` variable is dropped as part of this filter (the
       output field just contains ``total_precipitation`` and ``quality``).
       This can be controlled by settings the ``return_mask`` flag from
       ``False`` to ``True``.

    Notes
    -----

    The ``rodeo_opera_preprocessing`` filter was primarily designed to
    work with the 'OPERA Pan-European' Composites. It's likely these
    filters will be moved into a plugin in the near-future.

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
        return_mask: bool = False,
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
        return_mask: bool, optional
            Whether or not to return the mask
        """
        self.total_precipitation = total_precipitation
        self.quality = quality
        self.mask = mask
        self.max_total_precipitation = max_total_precipitation
        self.return_mask = return_mask

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
        total_precipitation_masked, quality_masked = mask_opera(
            tp=total_precipitation.to_numpy(), quality=quality.to_numpy(), mask=mask.to_numpy()
        )

        # 2nd - apply clipping
        total_precipitation_cleaned, quality_clipped = clip_opera(
            tp=total_precipitation_masked,
            quality=quality_masked,
            max_total_precipitation=self.max_total_precipitation,
        )

        yield self.new_field_from_numpy(
            total_precipitation_cleaned, template=total_precipitation, param=self.total_precipitation
        )
        yield self.new_field_from_numpy(quality_clipped, template=quality, param=self.quality)

        if self.return_mask:
            yield mask
