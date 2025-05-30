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
from earthkit.meteo.wind.array import polar_to_xy
from earthkit.meteo.wind.array import xy_to_polar

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching


class WindComponents(MatchingFieldsFilter):
    """A filter to convert wind speed and direction to U and V components,
    and back.
    """

    @matching(
        select="param",
        forward=("u_component", "v_component"),
        backward=("wind_speed", "wind_direction"),
    )
    def __init__(
        self,
        *,
        u_component: str = "u",
        v_component: str = "v",
        wind_speed: str = "ws",
        wind_direction: str = "wdir",
        convention: str = "meteo",
        radians: bool = False,
    ) -> None:
        """Initialize the WindComponents filter.

        Parameters
        ----------
        u_component : str, optional
            Name of the U component, by default "u".
        v_component : str, optional
            Name of the V component, by default "v".
        wind_speed : str, optional
            Name of the wind speed parameter, by default "ws".
        wind_direction : str, optional
            Name of the wind direction parameter, by default "wdir".
        convention : str, optional
            Convention to use for conversion, by default "meteo".
        radians : bool, optional
            Whether to use radians, by default False.
        """

        self.u_component = u_component
        self.v_component = v_component
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.convention = convention
        self.radians = radians

        assert not self.radians, "Radians not (yet) supported"

    def forward_transform(self, u_component: ekd.Field, v_component: ekd.Field) -> Iterator[ekd.Field]:
        """Convert U and V wind components to wind speed and direction.

        Parameters
        ----------
        u_component : ekd.Field
            The U component of the wind.
        v_component : ekd.Field
            The V component of the wind.

        Returns
        -------
        Iterator[ekd.Field]
            The wind speed field.
            The wind direction field.
        """

        speed, direction = xy_to_polar(
            u_component.to_numpy(),
            v_component.to_numpy(),
            convention=self.convention,
        )

        yield self.new_field_from_numpy(speed, template=u_component, param=self.wind_speed)
        yield self.new_field_from_numpy(direction, template=v_component, param=self.wind_direction)

    def backward_transform(self, wind_speed: ekd.Field, wind_direction: ekd.Field) -> Iterator[ekd.Field]:
        """Convert wind speed and direction to U and V components.

        Parameters
        ----------
        wind_speed : ekd.Field
            The wind speed field.
        wind_direction : ekd.Field
            The wind direction field.

        Returns
        -------
        Iterator[ekd.Field]
            The U component of the wind.
            The V component of the wind.
        """

        u, v = polar_to_xy(
            wind_speed.to_numpy(),
            wind_direction.to_numpy(),
            convention=self.convention,
        )

        yield self.new_field_from_numpy(u, template=wind_speed, param=self.u_component)
        yield self.new_field_from_numpy(v, template=wind_direction, param=self.v_component)


filter_registry.register("uv_to_ddff", WindComponents)
filter_registry.register("ddff_to_uv", WindComponents.reversed)
