# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd

from . import filter_registry
from .matching import MatchingFieldsFilter
from .matching import matching


class VerticalVelocity(MatchingFieldsFilter):
    """A filter to convert vertical wind speed expressed in m/s to vertical wind speed expressed in Pa/s using the hydrostatic hypothesis,
    and back.
    """

    @matching(
        match="param",
        forward=("w_component", "temperature", "humidity"),
        backward=("wz_component", "temperature", "humidity"),
    )
    def __init__(
        self,
        *,
        w_component="w",
        wz_component="wz",
        temperature="t",
        humidity="q",
    ):
        # wind speed in Pa/s
        self.w_component = w_component
        # wind speed in m/s
        self.wz_component = wz_component
        self.temperature = temperature
        self.humidity = humidity

    def forward_transform(self, w_component: ekd.Field, temperature: ekd.Field, humidity: ekd.Field) -> ekd.Field:
        """Pa/s to m/s"""

        level = float(w_component._metadata.get("levelist", None))
        # here the pressure gradient is assimilated to volumic weight : hydrostatic hypothesis
        rho = (100 * level) / (287 * temperature.to_numpy() * (1 + 0.61 * humidity.to_numpy()) + 1e-8)
        wz = (-1.0 / (rho * 9.80665 + 1e-8)) * w_component.to_numpy()

        yield self.new_field_from_numpy(wz, template=w_component, param=self.wz_component)

    def backward_transform(self, wz_component: ekd.Field, temperature: ekd.Field, humidity: ekd.Field) -> ekd.Field:
        """m/s to Pa/s"""

        level = float(wz_component._metadata.get("levelist", None))
        # here the pressure gradient is assimilated to volumic weight : hydrostatic hypothesis
        rho = (100 * level) / (287 * temperature.to_numpy() * (1 + 0.61 * humidity.to_numpy()) + 1e-8)
        w = -1.0 * rho * 9.80665 * wz_component.to_numpy()

        yield self.new_field_from_numpy(w, template=wz_component, param=self.w_component)


filter_registry.register("w_2_wz", VerticalVelocity)
filter_registry.register("wz_2_w", VerticalVelocity.reversed)
