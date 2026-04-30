# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import pandas as pd

from anemoi.transform.filter import Filter
from anemoi.transform.filters.tabular import filter_registry
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


@filter_registry.register("impute_nans_tabular")
class ImputeNaNs(Filter):
    """Fill NaN values in a DataFrame with a fixed or per-column value.

    The configuration can contain either a list of column names to be
    considered or a prefix string (`column_prefix` key), where all columns
    starting with this prefix will be considered. If neither are provided, then
    all columns will be considered.

    The fill value is set with the `value` key, which can be a scalar applied
    to all selected columns or a dict mapping column names to individual fill
    values.

    Examples
    --------
    Using one value for all columns
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - impute_nans:
              value: 0.0
              columns:
              - obsvalue_0
              - obsvalue_1

    Using a dictionary to specify values per column
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - impute_nans:
              value: {"obsvalue_0": 0.0, "obsvalue_1": 999.9}
              columns:
              - obsvalue_0
              - obsvalue_1
    """

    def __init__(self, *, value: float | dict, columns: list[str] | None = None, column_prefix: str | None = None):
        if bool(columns) and bool(column_prefix):
            raise ValueError("Either columns or column_prefix may be specified, but not both.")
        self.value = value
        self.columns = columns
        self.column_prefix = column_prefix

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        if self.columns:
            subset = list(self.columns)
            raise_if_df_missing_cols(obs_df, self.columns)
        elif self.column_prefix:
            subset = [col for col in obs_df.columns if col.startswith(self.column_prefix)]
            if not subset:
                raise ValueError(f"No columns starting with '{self.column_prefix}' found in DataFrame.")
        else:
            subset = None

        logging.info(f"Dropping rows with all NaN values on df with length: {len(obs_df)}")
        return obs_df.fillna(self.value)
