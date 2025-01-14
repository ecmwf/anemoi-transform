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


def mask_opera(tp, quality, mask):
    print("✅✅", quality)
    print("✅✅✅", tp)
    tp[quality == 75] = np.nan
    tp[mask >= 5] = np.nan
    return tp


@filter_registry.register("rodeo_opera_mask")
class RodeoOperaMask(SimpleFilter):
    """A filter to select only good quality data i nrodeo opera data."""

    def __init__(
        self,
        *,
        tp="tp",
        quality="quality",
        mask="mask",
        output="tp_masked",
    ):
        self.tp = tp
        self.quality = quality
        self.tp_masked = output
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
        raise NotImplementedError("RodeoOperaMask is not reversible")

    def forward_transform(self, tp, quality, mask):
        """Mask out rodeo Opear data"""

        tp_masked = mask_opera(tp=tp.to_numpy(), quality=quality.to_numpy(), mask=mask.to_numpy())

        yield self.new_field_from_numpy(tp_masked, template=tp, param=self.tp_masked)

    def backward_transform(self, tp):
        raise NotImplementedError("RodeoOperaMask is not reversible")
