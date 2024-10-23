# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from earthkit.meteo.wind.array import polar_to_xy
from earthkit.meteo.wind.array import xy_to_polar

from . import register_filter
from .base import SimpleFilter


class WindComponents(SimpleFilter):
    """A filter to convert wind speed and direction to U and V components,
    and back.
    """

    def __init__(
        self,
        *,
        u_component="u",
        v_component="v",
        wind_speed="ws",
        wind_direction="wdir",
        convention="meteo",
        radians=False,
    ):
        self.u_component = u_component
        self.v_component = v_component
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.convention = convention
        self.radians = radians

        assert not self.radians, "Radians not (yet) supported"

    def forward(self, data):
        return self._transform(
            data,
            self.forward_transform,
            self.u_component,
            self.v_component,
        )

    def backward(self, data):
        return self._transform(
            data,
            self.backward_transform,
            self.wind_speed,
            self.wind_direction,
        )

    def forward_transform(self, u, v):
        """U/V to DD/FF"""

        speed, direction = xy_to_polar(
            u.to_numpy(),
            v.to_numpy(),
            convention=self.convention,
        )

        yield self.new_field_from_numpy(speed, template=u, param=self.wind_speed)
        yield self.new_field_from_numpy(direction, template=v, param=self.wind_direction)

    def backward_transform(self, speed, direction):
        """DD/FF to U/V"""

        u, v = polar_to_xy(
            speed.to_numpy(),
            direction.to_numpy(),
            convention=self.convention,
        )

        yield self.new_field_from_numpy(u, template=speed, param=self.u_component)
        yield self.new_field_from_numpy(v, template=direction, param=self.v_component)


register_filter("uv_2_ddff", WindComponents)
register_filter("ddff_2_uv", WindComponents.reversed)
