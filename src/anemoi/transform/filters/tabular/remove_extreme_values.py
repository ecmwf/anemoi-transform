# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Literal

import pandas as pd

from anemoi.transform.filters.tabular import TabularFilter
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


class RemoveExtremeValues(TabularFilter, registry_name="remove_extreme_values"):
    """Remove extreme values from a DataFrame, either by dropping rows where any
    column contains an extreme value (beyond ±threshold), or by masking extreme
    values.

    The configuration must contain one of:
     - a list of column names to be considered (`columns` key); or
     - a prefix string (`column_prefix` key), where all columns starting with
       this prefix will be considered.

    The threshold corresponding to an extreme value can be configured using the
    `threshold` key (default: 1e10).

    The method to remove extreme values can be configured with the `method` key
    (default: "drop").

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - remove_extreme_values:
              columns:
              - obsvalue_1
              method: drop
              threshold: 1e9

    """

    def __init__(
        self,
        *,
        columns: list[str] | None = None,
        column_prefix: str | None = None,
        threshold: float = 1e10,
        method: Literal["mask", "drop"] = "drop",
    ):
        if method not in ("mask", "drop"):
            raise ValueError(f"Invalid method '{method}'. Must be either 'mask' or 'drop'.")
        if bool(columns) == bool(column_prefix):
            raise ValueError("Either columns or column_prefix must be specified, but not both.")
        self.method = method
        self.columns = columns
        self.column_prefix = column_prefix
        self.threshold = threshold

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        if self.columns:
            obs_cols = list(self.columns)
            raise_if_df_missing_cols(obs_df, self.columns)
        elif self.column_prefix:
            obs_cols = [col for col in obs_df.columns if col.startswith(self.column_prefix)]
            if not obs_cols:
                raise ValueError(f"No columns starting with '{self.column_prefix}' found in DataFrame.")
        obs_cols += ["latitude", "longitude"]

        verb = "Dropping" if self.method == "drop" else "Masking"
        logging.info(f"{verb} rows with extreme values beyond ±{self.threshold}")

        mask = obs_df[obs_cols].abs() > self.threshold
        if self.method == "drop":
            mask = mask.any(axis=1)
            obs_df = obs_df[~mask]
        elif self.method == "mask":
            obs_df[obs_cols] = obs_df[obs_cols].mask(mask)

        return obs_df
