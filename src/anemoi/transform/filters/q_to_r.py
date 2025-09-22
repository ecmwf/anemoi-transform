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
import earthkit.meteo.thermo.array as thermo

from anemoi.transform.filters import filter_registry

from .matching import MatchingFieldsFilter
from .matching import matching


class HumidityConversion(MatchingFieldsFilter):
    """A filter to convert specific humidity to relative humidity using standard thermodynamical formulas.

    This filter provides forward and backward transformations between specific humidity and relative humidity,
    given temperature and pressure information. It is designed to be used in data processing pipelines where
    conversion between these humidity representations is required.

    Notes
    -----
    For more information, see the :func:`relative_humidity_from_specific_humidity <earthkit.meteo.thermo.array.relative_humidity_from_specific_humidity>`
    function in the earthkit-meteo documentation.

    """

    @matching(
        select="param",
        forward=("humidity", "temperature"),
        backward=("relative_humidity", "temperature"),
    )
    def __init__(
        self,
        *,
        relative_humidity: str = "r",
        temperature: str = "t",
        humidity: str = "q",
        return_inputs: Literal["all", "none"] | list[str] = "all",
    ):
        """Initialize the HumidityConversion filter.

        Parameters
        ----------
        relative_humidity : str, optional
            Name of the humidity parameter, by default "q".
        temperature : str, optional
            Name of the temperature parameter, by default "t".
        humidity : str, optional
            Name of the humidity parameter, by default "q".
        return_inputs : Literal["all", "none"] | list[str], optional
            List of which filter inputs should be returned, by default "all"
        """
        self.return_inputs = return_inputs
        self.relative_humidity = relative_humidity
        self.temperature = temperature
        self.humidity = humidity

    def forward_transform(self, humidity: ekd.Field, temperature: ekd.Field) -> Iterator[ekd.Field]:
        """This will return the relative humidity along with temperature from specific humidity and temperature"""
        pressure = 100 * float(humidity.metadata("levelist"))
        rh = thermo.relative_humidity_from_specific_humidity(temperature.to_numpy(), humidity.to_numpy(), pressure)
        yield self.new_field_from_numpy(rh, template=humidity, param=self.relative_humidity)

    def backward_transform(self, relative_humidity: ekd.Field, temperature: ekd.Field) -> Iterator[ekd.Field]:
        """This will return specific humidity along with temperature from relative humidity and temperature"""
        pressure = 100 * float(temperature.metadata("levelist"))  # levels are measured in hectopascals
        q = thermo.specific_humidity_from_relative_humidity(
            temperature.to_numpy(), relative_humidity.to_numpy(), pressure
        )
        yield self.new_field_from_numpy(q, template=relative_humidity, param=self.humidity)


filter_registry.register("q_to_r", HumidityConversion)
filter_registry.register("r_to_q", HumidityConversion.reversed)
