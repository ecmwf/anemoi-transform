# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Literal

import earthkit.data as ekd
import numpy as np

from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import SingleFieldFilter
from anemoi.transform.filters import filter_registry

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


class ApplyMaskMixin:
    """A filter to mask variables using an external file.

    The values of every filtered fields are set to NaN when they are either:

    - equal to `mask_value` if provided, or
    - meeting a `threshold` condition if provided.

    The variable can then optionally be renamed by appending `_{rename}` to the original name.

    Examples
    --------

    .. code-block:: yaml

      input:
        pipe:
          - source: # Can be `mars`, `netcdf`, etc.
              param: ...
          - apply_mask:
              path: /path/to/mask_file.grib # E.g. a land-sea mask
              mask_value: 0 # Will set to NaN all values where mask == 0 (i.e. sea points)
              rename: masked # The new variable will be named `{param}_masked`

    And with a threshold:

    .. code-block:: yaml

        input:
          pipe:
            - source: # Can be `mars`, `netcdf`, etc.
               param: ...
            - apply_mask:
                path: /path/to/mask_file.nc # E.g. a land-sea mask
                threshold_operator: "<=" # Operator to use for thresholding
                threshold: 0.5 # Will set to NaN all values where mask <= 0.5

    Notes
    -----

    The mask file should contain a variable with the same grid as the input data.

    """

    def __init__(
        self,
        *,
        path: str,
        mask_value: float | None = None,
        threshold: float | None = None,
        threshold_operator: Literal["<", "<=", ">", ">=", "==", "!="] = ">",
        rename: str | None = None,
    ):
        """Initialize the MaskVariable filter.

        Parameters
        ----------
        path : str
            Path to the external file containing the mask.
        mask_value : float, optional
            Value to be used for masking, by default 1.
        threshold : float, optional
            Threshold value for masking, by default None.
        rename : str, optional
            New name for the masked variable, by default None.
        """

        mask = ekd.from_source("file", path)[0].to_numpy(flatten=True)

        if mask_value is None and threshold is None:
            raise ValueError("Either `mask_value` or `threshold` must be provided.")

        if threshold is not None:
            if threshold_operator not in OPERATORS:
                raise ValueError(
                    f"Invalid threshold operator: {threshold_operator}. "
                    f"Valid operators are: {', '.join(OPERATORS.keys())}."
                )
            self._mask = OPERATORS[threshold_operator](mask, threshold)
        else:
            self._mask = mask == mask_value

        self._rename = rename

    def mask(self, data: ekd.Field) -> ekd.Field:
        """Apply the forward transformation to the data.

        Parameters
        ----------
        data : ekd.Field
            Input data to be transformed.

        Returns
        -------
        ekd.Field
            Transformed data.
        """
        extra = {}

        values = data.to_numpy(flatten=True)
        values[self._mask] = np.nan

        if self._rename is not None:
            param = data.metadata("param")
            name = f"{param}_{self._rename}"
            extra["param"] = name

        return new_field_from_numpy(values, template=data, **extra)


@filter_registry.register("apply_mask")
class MaskAllVariables(ApplyMaskMixin):
    """A filter to mask all variables using an external file."""

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        result = []
        for field in data:
            masked_field = self.mask(field)
            result.append(masked_field)
        return new_fieldlist_from_list(result)


@filter_registry.register("apply_mask_to_param")
class ApplyMaskToParam(ApplyMaskMixin, SingleFieldFilter):
    """A filter to mask a specific variable using an external file."""

    required_inputs = ["param"]

    def __init__(
        self,
        *,
        param: str,
        path: str,
        mask_value: float | None = None,
        threshold: float | None = None,
        threshold_operator: (
            Literal["<"] | Literal["<="] | Literal[">"] | Literal[">="] | Literal["=="] | Literal["!="]
        ) = ">",
        rename: str | None = None,
    ):
        SingleFieldFilter.__init__(self, param=param)
        ApplyMaskMixin.__init__(
            self,
            path=path,
            mask_value=mask_value,
            threshold=threshold,
            threshold_operator=threshold_operator,
            rename=rename,
        )
        self.param = param

    def forward_select(self):
        return {"param": self.param}

    def forward_transform(self, param: ekd.Field) -> ekd.Field:
        return self.mask(param)
