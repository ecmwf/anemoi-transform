# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from . import filter_registry
from .base import SimpleFilter


@filter_registry.register("sum")
class Sum(SimpleFilter):
    """A filter to sum some parameters"""

    def __init__(
        self,
        *,
        formula,
    ):
        assert isinstance(formula, dict)
        assert len(formula) == 1
        self.name = list(formula.keys())[0]
        self.args = list(formula.values())[0]

    def forward(self, data):
        return self._transform(data, self.forward_transform, *self.args)

    def backward(self, data):
        raise NotImplementedError("Sum is not reversible")

    def forward_transform(self, *args):
        """Sum the fuel components to get the total fuel"""
        total = None
        for arg in args:
            if total is None:
                template = arg
                total = template.to_numpy()
            else:
                total += arg.to_numpy()

        yield self.new_field_from_numpy(total, template=template, param=self.name)

    def backward_transform(self, data):
        raise NotImplementedError("Sum is not reversible")
