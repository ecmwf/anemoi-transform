# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import earthkit.data as ekd
import numpy as np

from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
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


@filter_registry.register("apply_mask")
class MaskVariable(Filter):
    """A filter to mask variables using external file."""

    def __init__(
        self,
        *,
        path: str,
        mask_value: float = None,
        threshold: float = None,
        threshold_operator: str = ">",
        rename: str = None,
    ):
        """Initialize the MaskVariable filter.

        Parameters
        ----------
        path : str
            Path to the external file containing the mask.
        mask_value : int, optional
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

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Apply the forward transformation to the data.

        Parameters
        ----------
        data : ekd.FieldList
            Input data to be transformed.

        Returns
        -------
        ekd.FieldList
            Transformed data.
        """
        result = []
        extra = {}
        for field in data:

            values = field.to_numpy(flatten=True)
            values[self._mask] = np.nan

            if self._rename is not None:
                param = field.metadata("param")
                name = f"{param}_{self._rename}"
                extra["param"] = name

            result.append(new_field_from_numpy(values, template=field, **extra))

        return new_fieldlist_from_list(result)
