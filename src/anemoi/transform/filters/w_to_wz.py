# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections.abc import Iterator
from typing import Literal

import earthkit.data as ekd

from anemoi.transform.constants import g_gravitational_acceleration
from anemoi.transform.filters import filter_registry

from .matching import MatchingFieldsFilter
from .matching import matching


class VerticalVelocity(MatchingFieldsFilter):
    """A filter to convert vertical wind speed expressed in m/s to vertical wind speed expressed in Pa/s using the hydrostatic hypothesis,
    and back.

    This filter converts geometric vertical velocity (provided in
    m/s) to vertical velocity in pressure coordinates (Pa/s). This filter
    must follow a source that provides the vertical velocity, humidity and
    temperature. The hydrostatic hypothesis is used for this conversion.

    """

    @matching(
        select="param",
        forward=("vertical_velocity", "temperature", "humidity"),
        backward=("geometric_vertical_velocity", "temperature", "humidity"),
    )
    def __init__(
        self,
        *,
        vertical_velocity: str = "w",
        geometric_vertical_velocity: str = "wz",
        temperature: str = "t",
        humidity: str = "q",
        return_inputs: Literal["all", "none"] | list[str] = "all",
    ):
        """Initialize the VerticalVelocity filter.

        Parameters
        ----------
        vertical_velocity : str, optional
            Name of the W component, by default "w".
        geometric_vertical_velocity : str, optional
            Name of the Wz (in m/s) component, by default "wz".
        temperature : str, optional
            Name of the temperature parameter, by default "t".
        humidity : str, optional
            Name of the humidity parameter, by default "q".
        return_inputs : Literal["all", "none"] | list[str], optional
            list of which filter inputs should be returned, by default "all"
        """
        self.return_inputs = return_inputs
        # wind speed in Pa/s
        self.vertical_velocity = vertical_velocity
        # wind speed in m/s
        self.geometric_vertical_velocity = geometric_vertical_velocity
        self.temperature = temperature
        self.humidity = humidity

    def forward_transform(
        self,
        vertical_velocity: ekd.Field,
        temperature: ekd.Field,
        humidity: ekd.Field,
    ) -> Iterator[ekd.Field]:
        """Convert vertical wind speed from Pa/s to m/s.

        Parameters
        ----------
        vertical_velocity : ekd.Field
            The vertical wind speed in Pa/s.
        temperature : ekd.Field
            The temperature field.
        humidity : ekd.Field
            The humidity field.

        Returns
        -------
        Iterator[ekd.Field]
            The vertical wind speed in m/s.
        """

        level = float(vertical_velocity._metadata.get("levelist", None))
        # here the pressure gradient is assimilated to volumic weight : hydrostatic hypothesis
        rho = (100 * level) / (287 * temperature.to_numpy() * (1 + 0.61 * humidity.to_numpy()) + 1e-8)
        wz = (-1.0 / (rho * g_gravitational_acceleration + 1e-8)) * vertical_velocity.to_numpy()

        yield self.new_field_from_numpy(wz, template=vertical_velocity, param=self.geometric_vertical_velocity)

    def backward_transform(
        self, geometric_vertical_velocity: ekd.Field, temperature: ekd.Field, humidity: ekd.Field
    ) -> Iterator[ekd.Field]:
        """Convert vertical wind speed from m/s to Pa/s.

        Parameters
        ----------
        geometric_vertical_velocity : ekd.Field
            The vertical wind speed in m/s.
        temperature : ekd.Field
            The temperature field.
        humidity : ekd.Field
            The humidity field.

        Returns
        -------
        Iterator[ekd.Field]
            The vertical wind speed in Pa/s.
        """

        level = float(geometric_vertical_velocity._metadata.get("levelist", None))
        # here the pressure gradient is assimilated to volumic weight : hydrostatic hypothesis
        rho = (100 * level) / (287 * temperature.to_numpy() * (1 + 0.61 * humidity.to_numpy()) + 1e-8)
        w = -1.0 * rho * g_gravitational_acceleration * geometric_vertical_velocity.to_numpy()

        yield self.new_field_from_numpy(w, template=geometric_vertical_velocity, param=self.vertical_velocity)


filter_registry.register("w_to_wz", VerticalVelocity)
filter_registry.register("wz_to_w", VerticalVelocity.reversed)
