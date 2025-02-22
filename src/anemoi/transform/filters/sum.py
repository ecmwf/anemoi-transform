# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any
from typing import Generator

from . import filter_registry
from .base import SimpleFilter

LOG = logging.getLogger(__name__)


@filter_registry.register("sum")
class Sum(SimpleFilter):
    """A filter to sum some parameters."""

    def __init__(self, *, formula: dict) -> None:
        assert isinstance(formula, dict)
        assert len(formula) == 1
        self.name = list(formula.keys())[0]
        self.args = list(formula.values())[0]
        LOG.warning("Using the sum filter will be deprecated in the future. Please do not rely on it.")

    def forward(self, data: Any) -> Any:
        return self._transform(data, self.forward_transform, *self.args)

    def backward(self, data: Any) -> None:
        raise NotImplementedError("Sum is not reversible")

    def forward_transform(self, *args: Any) -> Generator[Any, None, None]:
        """Sum the fuel components to get the total fuel."""
        total = None
        for arg in args:
            if total is None:
                template = arg
                total = template.to_numpy()
            else:
                total += arg.to_numpy()

        yield self.new_field_from_numpy(total, template=template, param=self.name)

    def backward_transform(self, data: Any) -> None:
        raise NotImplementedError("Sum is not reversible")
