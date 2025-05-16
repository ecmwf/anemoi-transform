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

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching


class Rescale(MatchingFieldsFilter):
    """A filter to rescale a parameter from a scale and an offset, and back."""

    @matching(
        select="param",
        forward=("param",),
        backward=("param",),
    )
    def __init__(
        self,
        *,
        scale: float,
        offset: float,
        param: str,
    ) -> None:
        """Parameters
        -------------
        scale : float
            The scale factor.
        offset : float
            The offset value.
        param : str
            The parameter to be rescaled.
        """

        self.scale = scale
        self.offset = offset
        self.param = param

    def forward_transform(self, param: ekd.Field) -> Iterator[ekd.Field]:
        """Apply the forward transformation (x to ax+b).

        Parameters
        ----------
        param : ekd.Field
            The input data to be transformed.

        Returns
        -------
        Iterator[ekd.Field]
            A generator yielding the transformed data.
        """
        rescaled = param.to_numpy() * self.scale + self.offset
        yield self.new_field_from_numpy(rescaled, template=param, param=self.param)

    def backward_transform(self, param: ekd.Field) -> Iterator[ekd.Field]:
        """Apply the backward transformation (ax+b to x).

        Parameters
        ----------
        param : ekd.Field
            The input data to be transformed.

        Returns
        -------
        Iterator[ekd.Field]
            A generator yielding the transformed data.
        """
        descaled = (param.to_numpy() - self.offset) / self.scale
        yield self.new_field_from_numpy(descaled, template=param, param=self.param)


class Convert(Rescale):
    """A filter to convert a parameter in a given unit to another unit, and back."""

    def __init__(self, *, unit_in: str, unit_out: str, param: str) -> None:
        """Parameters
        -------------
        unit_in : str
            The input unit.
        unit_out : str
            The output unit.
        param : str
            The parameter to be converted.
        """
        from cfunits import Units

        u0 = Units(unit_in)
        u1 = Units(unit_out)
        x1, x2 = 0.0, 1.0
        y1, y2 = Units.conform([x1, x2], u0, u1)
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1

        super().__init__(scale=a, offset=b, param=param)


filter_registry.register("rescale", Rescale)
filter_registry.register("convert", Convert)
