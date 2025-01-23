# (C) Copyright 2025 Anemoi contributors.
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


@filter_registry.register("cos_sin_mean_wave_direction")
class CosSinWaveDirection(SimpleFilter):
    """A filter to convert wind speed and direction to U and V components,
    and back.
    """

    def __init__(
        self,
        *,
        mean_wave_direction="mwd",
        cos_mean_wave_direction="cos_mwd",
        sin_mean_wave_direction="sin_mwd",
        radians=False,
    ):
        self.mean_wave_direction = mean_wave_direction
        self.cos_mean_wave_direction = cos_mean_wave_direction
        self.sin_mean_wave_direction = sin_mean_wave_direction
        self.radians = radians

    def forward(self, data):
        return self._transform(
            data,
            self.forward_transform,
            self.mean_wave_direction,
        )

    def backward(self, data):
        raise NotImplementedError("Not implemented")

    def forward_transform(self, mwd):

        data = mwd.to_numpy()
        if not self.radians:
            data = np.deg2rad(data)

        yield self.new_field_from_numpy(np.cos(data), template=mwd, param=self.cos_mean_wave_direction)
        yield self.new_field_from_numpy(np.sin(data), template=mwd, param=self.sin_mean_wave_direction)

    def backward_transform(self, cos_mwd, sin_mwd):
        raise NotImplementedError("Not implemented")
