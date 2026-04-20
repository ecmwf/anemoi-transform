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

from anemoi.transform.fields import FieldSelection
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters.fields import filter_registry

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


@filter_registry.register("apply_mask_fields")
class MaskVariable(Filter):
    """A filter to mask variables using a mask from a file or from a field in the pipeline.

    The values of every filtered field are set to NaN when they are either:

    - equal to ``mask_value`` if provided, or
    - meeting a ``threshold`` condition if provided.

    The variable can then optionally be renamed by appending ``_{rename}`` to the original name.

    The mask can be provided either as an external file via ``path``, or as a
    parameter name already present in the pipeline via ``mask_param``. When
    ``mask_param`` is used, the mask field is consumed and removed from the
    output.

    The ``param`` keyword can be used to specify which variables to mask. If
    not provided, all variables will be masked. It can be a single variable
    or a list of variables.

    The ``return_mask`` keyword can be used to control whether the mask is returned
    or consumed (not returned) by the filter (default: False, i.e. consumed).
    This is only used when the mask is provided through the ``mask_param`` keyword.

    Examples
    --------

    Using an external file:

    .. code-block:: yaml

      input:
        pipe:
          - source: # Can be `mars`, `netcdf`, etc.
              param: ...
          - apply_mask:
              path: /path/to/mask_file.grib # E.g. a land-sea mask
              mask_value: 0 # Will set to NaN all values where mask == 0 (i.e. sea points)
              rename: masked # The new variable will be named `{param}_masked`

    Using a field from the pipeline:

    .. code-block:: yaml

      input:
        pipe:
          - source: # Can be `mars`, `netcdf`, etc.
              param:
              - sd
              - lsm
          - apply_mask:
              mask_param: lsm   # Use the lsm field from the pipeline as mask
              mask_value: 0     # Will set to NaN all values where lsm == 0
              return_mask: true # The mask will be returned by the filter

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
        path: str | None = None,
        mask_param: str | None = None,
        mask_value: float | None = None,
        threshold: float | None = None,
        threshold_operator: str = ">",
        rename: str | None = None,
        param: str | list[str] | None = None,
        return_mask: bool = False,
    ) -> None:
        self.path = path
        self.mask_param = mask_param
        self.mask_value = mask_value
        self.threshold = threshold
        self.threshold_operator = threshold_operator
        self.rename = rename
        self.param = param if not isinstance(param, str) else [param]
        self.return_mask = return_mask
        self.prepare_filter()
        self._forward_selection = FieldSelection(**self.forward_select())

    def prepare_filter(self):
        """Set up the MaskVariable filter."""

        if (self.path is None) == (self.mask_param is None):
            raise ValueError("Exactly one of `path` or `mask_param` must be provided.")

        if (self.mask_value is None) == (self.threshold is None):
            raise ValueError("Exactly one of `mask_value` or `threshold` must be provided.")

        if self.threshold is not None:
            if self.threshold_operator not in OPERATORS:
                raise ValueError(
                    f"Invalid threshold operator: {self.threshold_operator}. "
                    f"Valid operators are: {', '.join(OPERATORS.keys())}."
                )

        if self.path is not None:
            if self.path.endswith(".npy"):
                mask = np.load(self.path)
            else:
                mask = ekd.from_source("file", self.path)[0].to_numpy(flatten=True)
            self.mask = self._compute_mask(mask)

    def _compute_mask(self, mask_values: np.ndarray) -> np.ndarray:
        if self.threshold is not None:
            return OPERATORS[self.threshold_operator](mask_values, self.threshold)
        return mask_values == self.mask_value

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

        return new_field_from_numpy(values, template=field, **metadata)

    def _separate_mask_and_fields(self, fields: ekd.FieldList) -> tuple[np.ndarray, ekd.FieldList]:
        if self.mask_param is None:
            return self.mask, fields

        mask_field = None
        remaining = []
        for field in fields:
            is_mask_field = field.metadata("param") == self.mask_param
            if is_mask_field:
                if mask_field is None:
                    # store first instance of mask field
                    mask_field = field

                # drop mask if not returning
                if not self.return_mask:
                    continue
            remaining.append(field)

        if mask_field is None:
            raise ValueError(f"Mask parameter '{self.mask_param}' not found in input data.")

        mask = self._compute_mask(mask_field.to_numpy(flatten=True))

        fields = new_fieldlist_from_list(remaining)
        return mask, fields

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:
        """Apply the mask to the data.

        When ``mask_param`` is set, the mask field is extracted from the
        pipeline data and removed from the output.

        Parameters
        ----------
        fields : ekd.FieldList
            Input data to be transformed.

        Returns
        -------
        ekd.FieldList
            Transformed data with mask applied.
        """

        self.mask, fields = self._separate_mask_and_fields(fields)

        result = []
        for field in fields:
            if self._forward_selection.match(field):
                field = self.forward_transform(field)
            result.append(field)

        return new_fieldlist_from_list(result)
