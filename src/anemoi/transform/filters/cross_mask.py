# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import operator
from collections.abc import Iterator

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching


@filter_registry.register("cross_mask")
class CrossMask(MatchingFieldsFilter):
    """A filter to mask values of one variable based on another variable.

    This filter masks the values of a target variable based on the values
    of a reference variable.
    The masking can be done in both forward and backward directions, using
    the same logic. The user can specify the comparison operator and threshold
    for masking.
    """

    @matching(
        select="param",
        forward=["mask_param", "param"],
        backward=["mask_param", "param"],
    )
    def __init__(
        self,
        *,
        mask_param: str,
        param: str,
        comparison_value: float,
        masked_value: float,
        operator: str = "eq",
        atol: float = 1e-8,
    ) -> None:
        """Initialize the CrossMask filter.

        Parameters
        ----------
        mask_param : str
            Name of the reference variable used for masking.
        param : str
            Name of the target variable to be masked.
        comparison_value : float
            The value to compare the reference variable against.
        masked_value : float
            The value to assign to the target variable when the masking condition is met.
        operator : str, optional
            The comparison operator to use for masking, by default "eq" (equal).
            Supported operators are "gt", "lt", "ge", "le", "eq", and "ne".
        atol : float, optional
            Absolute tolerance for the "eq" and "ne" operators, by default 1e-8.
            This is used to determine if the reference variable is close enough to
            comparison_value for masking.
        """

        self.mask_param = mask_param
        self.param = param
        self.masked_value = masked_value
        self.comparison_value = comparison_value
        self.operator = operator
        self.atol = atol

    def mask(self, data: ekd.Field, mask: ekd.Field) -> ekd.Field:
        """Mask the target variable based on the reference variable.

        Parameters
        ----------
        data : ekd.Field
            The target variable to be masked.
        mask : ekd.Field
            The reference variable used for masking.

        Returns
        -------
        ekd.Field
            The masked target variable.
        """

        mask_data: np.ndarray = mask.to_numpy()
        data_array: np.ndarray = data.to_numpy()

        match self.operator:
            case "eq":
                mask_array: np.ndarray = np.isclose(mask_data, self.comparison_value, atol=self.atol, rtol=0.0)
            case "ne":
                mask_array: np.ndarray = ~np.isclose(mask_data, self.comparison_value, atol=self.atol, rtol=0.0)
            case _:
                op = {
                    "gt": operator.gt,
                    "lt": operator.lt,
                    "ge": operator.ge,
                    "le": operator.le,
                }.get(self.operator)

                if op is None:
                    raise ValueError(f"Unsupported operator: {self.operator}")
                mask_array: np.ndarray = op(mask_data, self.comparison_value)

        masked_data = np.where(mask_array, self.masked_value, data_array)

        return self.new_field_from_numpy(masked_data, template=data, param=data.metadata("param") or self.param)  # type: ignore

    def forward_transform(self, mask_param: ekd.Field, param: ekd.Field) -> Iterator[ekd.Field]:  # type: ignore
        """Mask the target variable based on the reference variable in the forward direction.

        Parameters
        ----------
        mask_param : ekd.Field
            The reference variable used for masking.
        param : ekd.Field
            The target variable to be masked.

        Returns
        -------
        Iterator[ekd.Field]
            The masked target variable, followed by the reference variable unchanged.
        """

        yield self.mask(param, mask_param)
        yield mask_param

    def backward_transform(self, mask_param: ekd.Field, param: ekd.Field) -> Iterator[ekd.Field]:  # type: ignore
        """Mask the target variable based on the reference variable in the backward direction.

        Parameters
        ----------
        mask_param : ekd.Field
            The reference variable used for masking.
        param : ekd.Field
            The target variable to be masked.

        Returns
        -------
        Iterator[ekd.Field]
            The masked target variable, followed by the reference variable unchanged.
        """

        yield self.mask(param, mask_param)
        yield mask_param
