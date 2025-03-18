# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd
import numpy as np

from . import filter_registry
from .matching import MatchingFieldsFilter
from .matching import matching


class HumidityConversion(MatchingFieldsFilter):
    """A filter to convert specific humidity to relative humidity with standard thermodynamical formulas.
    """

    @matching(
        match="param",
        forward=("temperature", "humidity"),
        backward=("rel_humidity", "temperature"),
    )
    def __init__(
        self,
        *,
        rel_humidity="r",
        temperature="t",
        humidity="q",
    ):
        
        self.relative_humidity = rel_humidity
        self.temperature = temperature
        self.humidity = humidity

    def forward_transform(self, temperature: ekd.Field, humidity: ekd.Field) -> ekd.Field:
        """
        This will return the relative humidity along with temperature from specific humidity and temperature
        """

        level = float(humidity._metadata.get("levelist", None))
        
        # here we follow Bolton, 1980 
        # https://journals.ametsoc.org/view/journals/mwre/108/7/1520-0493_1980_108_1046_tcoept_2_0_co_2.xml
        # with T in kelvins
                
        psat = 6.112 * np.exp(
            (17.67 * (temperature.to_numpy() - 273.15) )/ (temperature.to_numpy() - 29.65)
            )
        qsat = psat * 0.622 / (level * 100.0 - (1.0 - 0.622) * psat)

        rh = humidity.to_numpy() / (qsat + 1e-8)

        yield self.new_field_from_numpy(rh, template=humidity, param=self.rel_humidity)
        yield temperature

    def backward_transform(self, relative_humidity: ekd.Field, temperature: ekd.Field) -> ekd.Field:
        """ 
        This will return specific humidity along with temperature from relative humidity and temperature
        """

        level = float(relative_humidity._metadata.get("levelist", None))
        
        # here we follow Bolton, 1980 
        # https://journals.ametsoc.org/view/journals/mwre/108/7/1520-0493_1980_108_1046_tcoept_2_0_co_2.xml
        # with T in kelvins
                
        psat = 611.2 * np.exp(
            (17.67 * (temperature.to_numpy() - 273.15) )/ (temperature.to_numpy() - 29.65)
            )
        qsat = psat * 0.622 / (level * 100.0 - (1.0 - 0.622) * psat)

        q  = relative_humidity * qsat

        yield self.new_field_from_numpy(q, template=relative_humidity, param=self.humidity)
        yield temperature


filter_registry.register("q_2_r", HumidityConversion)
filter_registry.register("r_2_q", HumidityConversion.reversed)
