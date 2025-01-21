# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from . import filter_registry
from .base import SimpleFilter

NODATA = -9.999e06
UNDETECTED = -8.888e06

_NODATA = 1
_UNDETECTED = 2
_INF = 3

MAX_TP = 10000
MAX_QI = 1


def clip_opera(tp, quality):
    tp[tp < 0] = 0
    tp[tp >= MAX_TP] = MAX_TP
    quality[quality >= MAX_QI] = MAX_QI

    return tp, quality


def mask_opera(tp, quality, mask):
    print("✅✅", quality)
    print("✅✅✅", tp)

    # # RAW HDF5 DATA FILTERING
    # tp[quality == NODATA] = np.nan
    # tp[quality == UNDETECTED] = np.nan

    # GRIB2 ENCODED DATA FILTERING
    # !won't work until Pedro's fix to compute mask based on quality
    # quality grib2 just have nans no NODATA or UNDETECTED values
    tp[mask == _NODATA] = np.nan
    tp[mask == _UNDETECTED] = np.nan
    tp[mask == _INF] = np.nan

    return tp


@filter_registry.register("rodeo_opera_preprocessing")
class RodeoOperaPreProcessing(SimpleFilter):
    """A filter to select only good quality data i nrodeo opera data."""

    def __init__(
        self,
        *,
        tp="tp",
        quality="quality",
        mask="mask",
        output="tp_cleaned",
    ):
        self.tp = tp
        self.quality = quality
        self.tp_cleaned = output
        self.mask = mask

    def forward(self, data):
        return self._transform(
            data,
            self.forward_transform,
            self.tp,
            self.quality,
            self.mask,
        )

    def backward(self, data):
        raise NotImplementedError("RodeoOperaPreProcessing is not reversible")

    def forward_transform(self, tp, quality, mask):
        """Pre-process Rodeo Opera data"""

        # 1st - apply masking
        tp_masked = mask_opera(tp=tp.to_numpy(), quality=quality.to_numpy(), mask=mask.to_numpy())

        # 2nd - apply clipping
        tp_cleaned, quality = clip_opera(tp=tp_masked, quality=quality.to_numpy())

        yield self.new_field_from_numpy(tp_cleaned, template=tp, param=self.tp_cleaned)

    def backward_transform(self, tp):
        raise NotImplementedError("RodeoOperaPreProcessing is not reversible")
