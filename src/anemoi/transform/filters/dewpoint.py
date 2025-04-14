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


class DewPoint(MatchingFieldsFilter):
    """A filter to extract dewpoint temperature from relative humidity and temperature
    """

    @matching(
        select="param",
        forward=("temperature", "relative_humidity"),
    )
    def __init__(
        self,
        *,
        relative_humidity="r",
        temperature="t",
        dewpoint="d"
     ):
        
        self.relative_humidity = relative_humidity
        self.temperature = temperature
        self.dewpoint = dewpoint

    def forward_transform(self, temperature: ekd.Field, relative_humidity: ekd.Field) -> ekd.Field:
        """
        Return the dewpoint temperature (Td, in K) along with temperature (K) and relative humidity (in %)
        """
        
        # here we follow Bolton, 1980 
        # https://journals.ametsoc.org/view/journals/mwre/108/7/1520-0493_1980_108_1046_tcoept_2_0_co_2.xml
        # with T in kelvins
                
        pvap_ratio = np.clip(
            (relative_humidity.to_numpy() / 100.0) * np.exp(
            (17.67 * (temperature.to_numpy() - 273.15) ) / (temperature.to_numpy() - 29.65)
            ),
            a_min=1e-8, a_max=None
            )
        
        td = 273.15 + 243.5 * np.log(pvap_ratio) / (17.67 - np.log(pvap_ratio)) 

        yield self.new_field_from_numpy(td, template=temperature, param=self.dewpoint)
        yield temperature
        yield relative_humidity

    def backward_transform(self, dewpoint: ekd.Field, temperature: ekd.Field) -> ekd.Field:
        """
        This will return the relative humidity (in %) from temperature (in K) and dewpoint (Td, in K),
        along with temperature and dewpoint
        """
        
        # here we follow Bolton, 1980 
        # https://journals.ametsoc.org/view/journals/mwre/108/7/1520-0493_1980_108_1046_tcoept_2_0_co_2.xml
        # with T in kelvins
                        
        psat_ratio = np.exp(
            (17.67 * (temperature.to_numpy() - 273.15) )/ (temperature.to_numpy() - 29.65)
            )
        
        t_norm = (dewpoint.to_numpy() - 273.15) / 243.5
        
        pvap_ratio = np.exp(
            17.67 * t_norm / (1.0 + t_norm)
        )
        
        rh = 100.0 * pvap_ratio / (psat_ratio + 1e-8)

        yield self.new_field_from_numpy(rh, template=temperature, param=self.relative_humidity)
        yield dewpoint
        yield temperature

filter_registry.register("r-2-d", DewPoint)
filter_registry.register("d-2-r", DewPoint.reversed)
