# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd
from earthkit.meteo import thermo
import numpy as np

from . import filter_registry
from .matching import MatchingFieldsFilter
from .matching import matching


class HumidityConversion(MatchingFieldsFilter):
    """A filter to convert specific humidity to relative humidity with standard thermodynamical formulas.
    """

    @matching(
        select="param",
        forward=("temperature", "humidity"),
        backward=("relative_humidity", "temperature"),
    )
    def __init__(
        self,
        *,
        relative_humidity="r",
        temperature="t",
        humidity="q",
    ):
        
        self.relative_humidity = relative_humidity
        self.temperature = temperature
        self.humidity = humidity

    def forward_transform(self, humidity: ekd.Field, temperature: ekd.Field) -> ekd.Field:
        """
        This will return the relative humidity along with temperature from specific humidity and temperature
        """
        pressure = 100 * float(humidity._metadata.get("levelist", None)) # levels are measured in hectopascals
        rh = thermo.relative_humidity_from_specific_humidity(temperature.to_numpy(),humidity.to_numpy(),pressure)

        yield self.new_field_from_numpy(rh, template=humidity, param=self.relative_humidity)
        yield temperature
        yield humidity

    def backward_transform(self, relative_humidity: ekd.Field, temperature: ekd.Field) -> ekd.Field:
        """ 
        This will return specific humidity along with temperature from relative humidity and temperature
        """
        pressure = 100 * float(temperature._metadata.get("levelist", None)) # levels are measured in hectopascals
    
        q  = thermo.specific_humidity_from_relative_humidity(temperature.to_numpy(),relative_humidity.to_numpy(),pressure)

        yield self.new_field_from_numpy(q, template=relative_humidity, param=self.humidity)
        yield temperature
        yield relative_humidity

filter_registry.register("q_2_r", HumidityConversion)
filter_registry.register("r_2_q", HumidityConversion.reversed)
