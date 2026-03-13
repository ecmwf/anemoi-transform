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


@filter_registry.register("apply_mask")
class MaskVariable(SingleFieldFilter):
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

    # path: str
    required_inputs = ("path",)
    # mask_value: float | None,
    # threshold: float | None,
    # threshold_operator: Literal["<", "<=", ">", ">=", "==", "!="],
    # rename: str | None,
    # param: str | None,
    optional_inputs = {
        "mask_value": None,
        "threshold": None,
        "threshold_operator": ">",
        "rename": None,
        "param": None,
    }

    def prepare_filter(self):
        """Setup the MaskVariable filter.

        Note:
        path : str
            Path to the external file containing the mask.
        mask_value : int, optional
            Value to be used for masking, by default 1.
        threshold : float, optional
            Threshold value for masking, by default None.
        rename : str, optional
            New name for the masked variable, by default None.
        """

        mask = ekd.from_source("file", self.path)[0].to_numpy(flatten=True)

        if self.mask_value is None and self.threshold is None:
            raise ValueError("Either `mask_value` or `threshold` must be provided.")

        if self.threshold is not None:
            if self.threshold_operator not in OPERATORS:
                raise ValueError(
                    f"Invalid threshold operator: {self.threshold_operator}. "
                    f"Valid operators are: {', '.join(OPERATORS.keys())}."
                )
            self.mask = OPERATORS[self.threshold_operator](mask, self.threshold)
        else:
            self.mask = mask == self.mask_value

    def forward_select(self):
        if self.param is not None:
            return {"param": self.param}
        return {}

    def forward_transform(self, field: ekd.Field) -> ekd.Field:
        """Apply the forward transformation to the field.

        Parameters
        ----------
        field : ekd.Field
            Input field to be transformed.

        Returns
        -------
        ekd.Field
            Transformed field.
        """
        metadata = {}
        values = field.to_numpy(flatten=True)
        values[self.mask] = np.nan

        if self.rename is not None:
            param = field.metadata("param")
            name = f"{param}_{self.rename}"
            metadata["param"] = name

        return self.new_field_from_numpy(values, template=field, **metadata)
