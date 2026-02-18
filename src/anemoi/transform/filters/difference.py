# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Literal

import earthkit.data as ekd

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter, matching


@filter_registry.register("difference")
class DifferenceFilter(MatchingFieldsFilter):
    """A filter to compute a simple difference between two variables.

    This filter converts two fields X and Y to a field representing the difference between the two (X - Y).

    """

    @matching(
        select="param",
        forward=("x", "y"),
        backward=("diff",)
    )
    def __init__(
        self,
        *,
        x: str,
        y: str,
        diff: str,
        return_inputs: Literal["all", "none"] | list[str] = "all",
    ) -> None:
        """Initialize the Diff filter.

        Parameters
        ----------
        x : str
            Name of the first variable.
        y : str
            Name of the second variable.
        diff : str
            Name of the difference variable.
        """

        self.x = x
        self.y = y
        self.diff = diff
        self.return_inputs = return_inputs


    def forward_transform(self, x: ekd.Field, y: ekd.Field) -> ekd.Field:
        """Compute the difference between two fields (X - Y).

        Parameters
        ----------
        x : ekd.Field
            The first variable.
        y : ekd.Field
            The second variable.

        Returns
        -------
        ekd.Field
            The difference field.
        """

        diff = x.to_numpy() - y.to_numpy()
        yield self.new_field_from_numpy(diff, template=x, param=self.diff)

    def backward(self, diff: ekd.FieldList) -> ekd.FieldList:
        raise NotImplementedError("Difference filter is not reversible")


