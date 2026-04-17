# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections.abc import Iterator

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filters.fields import filter_registry
from anemoi.transform.filters.fields.matching import MatchingFieldsFilter
from anemoi.transform.filters.fields.matching import matching

LOG = logging.getLogger(__name__)

OPERATORS = {
    ">": np.greater,
    "<": np.less,
    "==": np.equal,
    "!=": np.not_equal,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "gt": np.greater,
    "lt": np.less,
    "eq": np.equal,
    "ne": np.not_equal,
    "ge": np.greater_equal,
    "le": np.less_equal,
}


class MaskWithField(MatchingFieldsFilter):
    """A filter to mask fields using another field as a mask.

    The values of the selected field are set to NaN based on the values of the mask field.

    Examples
    --------

    .. code-block:: yaml

      - mask_with_field:
          mask_param: lsm
          param: tp
          mask_value: 0 # Will set to NaN all values where lsm == 0
          rename: masked # The new variable will be named `tp_masked`

    """

    @matching(
        select="param",
        forward=("mask_param", "param"),
        return_inputs="all",
    )
    def __init__(
        self,
        *,
        mask_param: str,
        param: str,
        mask_value: float | None = None,
        threshold: float | None = None,
        threshold_operator: str = ">",
        rename: str | None = None,
        drop_mask: bool = False,
    ) -> None:
        """Initialize the MaskWithField filter.

        Parameters
        ----------
        mask_param : str
            Name of the parameter to use as a mask.
        param : str
            Name of the parameter to apply the mask to.
        mask_value : float, optional
            Value to be used for masking, by default None.
        threshold : float, optional
            Threshold value for masking, by default None.
        threshold_operator : str, optional
            Operator to use for thresholding, by default ">".
        rename : str, optional
            New name for the masked variable, by default None.
        drop_mask : bool, optional
            Whether to drop the mask field from the output, by default False.
        """
        self.mask_param = mask_param
        self.param = param
        self.mask_value = mask_value
        self.threshold = threshold
        self.threshold_operator = threshold_operator
        self.rename = rename
        self.drop_mask = drop_mask

        if self.drop_mask:
            self.return_inputs = ["param"]

        if self.mask_value is None and self.threshold is None:
            raise ValueError("Either `mask_value` or `threshold` must be provided.")

        if self.threshold is not None and self.threshold_operator not in OPERATORS:
            raise ValueError(
                f"Invalid threshold operator: {self.threshold_operator}. "
                f"Valid operators are: {', '.join(OPERATORS.keys())}."
            )

    def forward_transform(self, mask_param: ekd.Field, param: ekd.Field) -> Iterator[ekd.Field]:
        """Apply the mask to the field.

        Parameters
        ----------
        mask_param : ekd.Field
            The field to use as a mask.
        param : ekd.Field
            The field to apply the mask to.

        Returns
        -------
        Iterator[ekd.Field]
            The masked field.
        """
        mask_values = mask_param.to_numpy(flatten=True)
        values = param.to_numpy(flatten=True)

        if self.threshold is not None:
            mask = OPERATORS[self.threshold_operator](mask_values, self.threshold)
        else:
            mask = mask_values == self.mask_value

        values[mask] = np.nan

        param_name = self.param
        if self.rename is not None:
            param_name = f"{self.param}_{self.rename}"

        yield self.new_field_from_numpy(values, template=param, param=param_name)


filter_registry.register("mask_with_field", MaskWithField)
