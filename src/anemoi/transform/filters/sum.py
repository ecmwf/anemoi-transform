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
from typing import Iterator

import earthkit.data as ekd

from . import filter_registry
from .base import SimpleFilter

LOG = logging.getLogger(__name__)


@filter_registry.register("sum")
class Sum(SimpleFilter):
    """A filter to sum some parameters."""

    def __init__(self, *, formula: dict) -> None:
        """Initialize the Sum filter.

        Parameters
        ----------
        formula : dict
            Dictionary containing the formula for summing parameters.
        """
        assert isinstance(formula, dict)
        assert len(formula) == 1
        self.name = list(formula.keys())[0]
        self.args = list(formula.values())[0]
        LOG.warning("Using the sum filter will be deprecated in the future. Please do not rely on it.")

    def forward(self, data: Any) -> Any:
        """Apply the forward transformation to the data.

        Parameters
        ----------
        data : Any
            Input data to be transformed.

        Returns
        -------
        Any
            Transformed data.
        """
        return self._transform(data, self.forward_transform, *self.args)

    def backward(self, data: Any) -> None:
        """Raise an error as Sum is not reversible.

        Parameters
        ----------
        data : Any
            Input data to be transformed.

        Raises
        ------
        NotImplementedError
            Always raised as this operation is not supported.
        """
        raise NotImplementedError("Sum is not reversible")

    def forward_transform(self, *args: Any) -> Iterator[ekd.Field]:
        """Sum the fuel components to get the total fuel.

        Parameters
        ----------
        args : Any
            Fuel components to be summed.

        Yields
        ------
        Any
            Total fuel field.
        """
        total = None
        for arg in args:
            if total is None:
                template = arg
                total = template.to_numpy()
            else:
                total += arg.to_numpy()

        yield self.new_field_from_numpy(total, template=template, param=self.name)

    def backward_transform(self, data: Any) -> None:
        """Raise an error as Sum is not reversible.

        Parameters
        ----------
        data : Any
            Input data to be transformed.

        Raises
        ------
        NotImplementedError
            Always raised as this operation is not supported.
        """
        raise NotImplementedError("Sum is not reversible")
