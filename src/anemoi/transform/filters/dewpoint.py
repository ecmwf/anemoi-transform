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

from . import filter_registry
from .matching import MatchingFieldsFilter
from .matching import matching


class DewPoint(MatchingFieldsFilter):
    """A filter to extract dewpoint temperature from relative humidity and temperature"""

    @matching(
        select="param",
        forward=("temperature", "relative_humidity"),
    )
    def __init__(self, *, relative_humidity="r", temperature="t", dewpoint="d"):

        self.relative_humidity = relative_humidity
        self.temperature = temperature
        self.dewpoint = dewpoint

    def forward_transform(self, temperature: ekd.Field, relative_humidity: ekd.Field) -> ekd.Field:
        """Return the dewpoint temperature (Td, in K) along with temperature (K) and relative humidity (in %)"""

        td = thermo.dewpoint_from_relative_humidity(temperature.to_numpy(), relative_humidity.to_numpy())

        yield self.new_field_from_numpy(td, template=temperature, param=self.dewpoint)
        yield temperature
        yield relative_humidity

    def backward_transform(self, dewpoint: ekd.Field, temperature: ekd.Field) -> ekd.Field:
        """This will return the relative humidity (in %) from temperature (in K) and dewpoint (Td, in K),
        along with temperature and dewpoint
        """
        rh = thermo.relative_humidity_from_dewpoint(temperature.to_numpy(), dewpoint.to_numpy())

        yield self.new_field_from_numpy(rh, template=temperature, param=self.relative_humidity)
        yield dewpoint
        yield temperature


filter_registry.register("r_2_d", DewPoint)
filter_registry.register("d_2_r", DewPoint.reversed)
