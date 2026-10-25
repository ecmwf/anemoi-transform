# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Iterator
from typing import Literal

import earthkit.meteo.thermo.array as thermo

from anemoi.transform import Field
from anemoi.transform.filters.fields import filter_registry

from .matching import MatchingFieldsFilter
from .matching import MatchingSpec


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

    MATCHING = MatchingSpec(
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
        super().__init__()

    def forward_transform(self, humidity: Field, temperature: Field) -> Iterator[Field]:
        """This will return the relative humidity along with temperature from specific humidity and temperature"""
        humidity.check_units("kg kg**-1")
        temperature.check_units("K")
        pressure = 100 * float(humidity.vertical.level())
        rh = thermo.relative_humidity_from_specific_humidity(temperature.to_numpy(), humidity.to_numpy(), pressure)
        yield Field.from_numpy(
            rh,
            template=humidity,
            parameter={
                "variable": self.relative_humidity,
                "units": "%",
            },
        )

    def backward_transform(self, relative_humidity: Field, temperature: Field) -> Iterator[Field]:
        """This will return specific humidity along with temperature from relative humidity and temperature"""
        relative_humidity.check_units("%")
        temperature.check_units("K")
        pressure = 100 * float(temperature.vertical.level())  # levels are measured in hectopascals
        q = thermo.specific_humidity_from_relative_humidity(
            temperature.to_numpy(), relative_humidity.to_numpy(), pressure
        )
        yield Field.from_numpy(
            q,
            template=relative_humidity,
            parameter={
                "variable": self.humidity,
                "units": "kg kg**-1",
            },
        )


filter_registry.register("q_to_r", HumidityConversion)
filter_registry.register("r_to_q", HumidityConversion.reversed)
