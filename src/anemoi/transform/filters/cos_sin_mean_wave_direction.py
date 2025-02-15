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
    """A filter to convert mean wave direction to cos() and sin()
    and back.
    """

    def __init__(
        self,
        *,
        mean_wave_direction="mwd",
        cos_mean_wave_direction="cos_mwd",
        sin_mean_wave_direction="sin_mwd",
    ):
        self.mean_wave_direction = mean_wave_direction
        self.cos_mean_wave_direction = cos_mean_wave_direction
        self.sin_mean_wave_direction = sin_mean_wave_direction

    def forward(self, data):
        return self._transform(
            data,
            self.forward_transform,
            self.mean_wave_direction,
        )

    def backward(self, data):
        return self._transform(
            data,
            self.backward_transform,
            self.cos_mean_wave_direction,
            self.sin_mean_wave_direction,
        )

    def forward_transform(self, mwd):

        data = mwd.to_numpy()
        data = np.deg2rad(data)

        yield self.new_field_from_numpy(np.cos(data), template=mwd, param=self.cos_mean_wave_direction)
        yield self.new_field_from_numpy(np.sin(data), template=mwd, param=self.sin_mean_wave_direction)

    def backward_transform(self, cos_mwd, sin_mwd):

        mwd = np.rad2deg(np.arctan2(sin_mwd.to_numpy(), cos_mwd.to_numpy()))
        mwd = np.where(mwd >= 360, mwd - 360, mwd)
        mwd = np.where(mwd < 0, mwd + 360, mwd)

        yield self.new_field_from_numpy(mwd, template=cos_mwd, param=self.mean_wave_direction)

    def patch_data_request(self, data_request):
        """We have a chance to modify the data request here."""

        param = data_request.get("param")
        if param is None:
            return data_request

        if self.cos_mean_wave_direction in param or self.sin_mean_wave_direction in param:
            data_request["param"] = [
                p for p in param if p not in (self.cos_mean_wave_direction, self.sin_mean_wave_direction)
            ]
            data_request["param"].append(self.mean_wave_direction)

        return data_request
