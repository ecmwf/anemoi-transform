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


class DropNaNs(TabularFilter, registry_name="drop_nans"):
    """Drop rows from a DataFrame where any/all selected columns are NaN.

    The configuration can contain either a list of column names to be
    considered or a prefix string (`column_prefix` key), where all columns
    starting with this prefix will be considered. If neither are provided, then
    all columns will be considered.

    The criterion for dropping a row can be set with the `how` key, either to
    'any' (default) or 'all', determining whether any or all selected columns
    must contain NaN values for a row to be dropped.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - drop_nans:
              how: any
              columns:
              - obsvalue_0
              - obsvalue_1

    """

    def __init__(
        self, *, how: Literal["any", "all"] = "any", columns: list[str] | None = None, column_prefix: str | None = None
    ):
        if how not in ("any", "all"):
            raise ValueError(f"DropNaNs - 'how' must be either 'any' or 'all', not '{how}'.")
        if bool(columns) and bool(column_prefix):
            raise ValueError("Either columns or column_prefix may be specified, but not both.")
        self.how = how
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
        return obs_df.dropna(how=self.how, subset=subset)
