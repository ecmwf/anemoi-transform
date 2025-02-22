# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from typing import Generator

from . import filter_registry
from .base import SimpleFilter


class Rescale(SimpleFilter):
    """A filter to rescale a parameter from a scale and an offset, and back."""

    def __init__(
        self,
        *,
        scale: float,
        offset: float,
        param: str,
    ) -> None:
        self.scale = scale
        self.offset = offset
        self.param = param

    def forward(self, data: Any) -> Any:
        return self._transform(data, self.forward_transform, self.param)

    def backward(self, data: Any) -> Any:
        return self._transform(
            data,
            self.backward_transform,
            self.param,
        )

    def forward_transform(self, x: Any) -> Generator[Any, None, None]:
        """X to ax+b."""

        rescaled = x.to_numpy() * self.scale + self.offset

        yield self.new_field_from_numpy(rescaled, template=x, param=self.param)

    def backward_transform(self, x: Any) -> Generator[Any, None, None]:
        """Ax+b to x."""

        descaled = (x.to_numpy() - self.offset) / self.scale

        yield self.new_field_from_numpy(descaled, template=x, param=self.param)


class Convert(Rescale):
    """A filter to convert a parameter in a given unit to another unit, and back."""

    def __init__(self, *, unit_in: str, unit_out: str, param: str) -> None:
        from cfunits import Units

        u0 = Units(unit_in)
        u1 = Units(unit_out)
        x1, x2 = 0.0, 1.0
        y1, y2 = Units.conform([x1, x2], u0, u1)
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        self.scale = a
        self.offset = b
        self.param = param


filter_registry.register("rescale", Rescale)
filter_registry.register("convert", Convert)
