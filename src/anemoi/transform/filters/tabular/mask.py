# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np
import pandas as pd

from anemoi.transform.filter import Filter
from anemoi.transform.filters.tabular import filter_registry
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols

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


@filter_registry.register("mask_tabular")
class MaskValues(Filter):
    """Mask the columns of a DataFrame based on a condition.

    The configuration should be a dictionary with column names as keys and
    the mask condition as a dictionary with "value" and optionally "operator" keys.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - mask:
              foo:
                value: 2
              bar:
                value: 0.5
                operator: ">"

    """

    def __init__(self, **config):
        if not config:
            raise ValueError("No columns to mask were specified.")
        self.config = {}
        for col, condition in config.items():
            if not isinstance(condition, dict):
                raise ValueError(f"Mask condition for column {col} must be a dictionary, ")

            if "value" not in condition:
                raise ValueError(f"Mask condition for column {col} must contain a 'value' key.")

            operator_str = condition.get("operator", "==")
            if operator_str not in OPERATORS:
                raise ValueError(
                    f"Invalid operator '{operator_str}' for column {col}. "
                    f"Valid operators are: {', '.join(OPERATORS.keys())}."
                )
            self.config[col] = {"value": condition["value"], "operator": OPERATORS[operator_str]}

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        raise_if_df_missing_cols(obs_df, self.config.keys())

        for col, condition in self.config.items():
            mask_value = condition["value"]
            operator = condition["operator"]
            logging.info(f"Masking {col} where values {operator.__name__} {mask_value}")
            obs_df[col] = obs_df[col].mask(operator(obs_df[col], mask_value))
        return obs_df
