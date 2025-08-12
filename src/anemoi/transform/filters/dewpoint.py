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
from earthkit.meteo import thermo

from anemoi.transform.filters import filter_registry

from .matching import MatchingFieldsFilter
from .matching import matching

EPS = 1.0e-4


class DewPoint(MatchingFieldsFilter):
    """A filter to extract dewpoint temperature from relative humidity and temperature"""

    @matching(
        select="param",
        forward=("relative_humidity", "temperature"),
        backward=("dewpoint", "temperature"),
    )
    def __init__(
        self,
        *,
        relative_humidity: str = "r",
        temperature: str = "t",
        dewpoint: str = "d",
        return_inputs: Literal["all", "none"] | list[str] = "all",
    ):
        """Initialize the DewPoint filter.

        Parameters
        ----------
        relative_humidity : str, optional
            Name of the humidity parameter, by default "r".
        temperature : str, optional
            Name of the temperature parameter, by default "t".
        return_inputs : Literal["all", "none"] | list[str], optional
            List of which filter inputs should be returned, by default "all"
        """
        self.return_inputs = return_inputs
        self.relative_humidity = relative_humidity
        self.temperature = temperature
        self.dewpoint = dewpoint

    def forward_transform(self, relative_humidity: ekd.Field, temperature: ekd.Field) -> Iterator[ekd.Field]:
        """Return the dewpoint temperature (Td, in K) along with temperature (K) and relative humidity (in %)"""

        relative_humidity_values = relative_humidity.to_numpy()
        relative_humidity_values[relative_humidity_values == 0] = EPS
        td = thermo.dewpoint_from_relative_humidity(t=temperature.to_numpy(), r=relative_humidity_values)
        yield self.new_field_from_numpy(td, template=relative_humidity, param=self.dewpoint)

    def backward_transform(self, dewpoint: ekd.Field, temperature: ekd.Field) -> Iterator[ekd.Field]:
        """This will return the relative humidity (in %) from temperature (in K) and dewpoint (Td, in K),
        along with temperature and dewpoint
        """
        rh = thermo.relative_humidity_from_dewpoint(t=temperature.to_numpy(), td=dewpoint.to_numpy())
        yield self.new_field_from_numpy(rh, template=temperature, param=self.relative_humidity)


filter_registry.register("r_to_d", DewPoint)
filter_registry.register("d_to_r", DewPoint.reversed)
