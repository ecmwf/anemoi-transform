# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from abc import ABC
from abc import abstractmethod
from typing import Callable

import earthkit.data as ekd

from anemoi.transform.filter import SingleFieldFilter
from anemoi.transform.filters import filter_registry


class Rescaler:
    def __init__(self, scale, offset):
        self.scale = scale
        self.offset = offset

    def forward(self, x):
        return x * self.scale + self.offset

    def backward(self, x):
        return (x - self.offset) / self.scale


class RescaleMixin(ABC):
    # inheriting classes should define required_inputs (which must include param)
    # and must define self.rescaler in prepare_filter
    param: str
    rescaler: Rescaler
    # intended to be inherited from SingleFieldFilter
    new_field_from_numpy: Callable

    @abstractmethod
    def prepare_filter(self):
        raise NotImplementedError("prepare_filter must be implemented by subclasses.")

    def forward_select(self):
        return {"param": self.param}

    def forward_transform(self, param: ekd.Field) -> ekd.Field:
        """Apply the forward transformation (x to ax+b)."""
        rescaled = self.rescaler.forward(param.to_numpy())
        return self.new_field_from_numpy(rescaled, template=param, param=self.param)

    def backward_transform(self, param: ekd.Field) -> ekd.Field:
        """Apply the backward transformation (ax+b to x)."""
        descaled = self.rescaler.backward(param.to_numpy())
        return self.new_field_from_numpy(descaled, template=param, param=self.param)


class Rescale(RescaleMixin, SingleFieldFilter):
    """A filter to rescale a parameter from a scale and an offset, and back."""

    required_inputs = ("scale", "offset", "param")

    def prepare_filter(self):
        self.rescaler = Rescaler(self.scale, self.offset)


class Convert(RescaleMixin, SingleFieldFilter):
    """A filter to convert a parameter in a given unit to another unit, and back.

    This filter uses :mod:`cfunits` (see the `cfunits documentation <https://ncas-cms.github.io/cfunits/>`_)
    to compute the scale and offset.

    Examples
    --------

    .. code-block:: yaml

      input:
        pipe:
          - necdf:
              path: /path/to/file.nc
              variables: [t2m, orog]
          - rescale:
              unit_in: degC
              unit_out: K
              param: t2m # The parameter to be converted

    """

    required_inputs = ("unit_in", "unit_out", "param")

    def prepare_filter(self):
        from cfunits import Units

        u0 = Units(self.unit_in)
        u1 = Units(self.unit_out)
        x1, x2 = 0.0, 1.0
        y1, y2 = Units.conform([x1, x2], u0, u1)
        scale = (y2 - y1) / (x2 - x1)
        offset = y1 - scale * x1

        self.rescaler = Rescaler(scale, offset)


filter_registry.register("rescale", Rescale)
filter_registry.register("convert", Convert)
