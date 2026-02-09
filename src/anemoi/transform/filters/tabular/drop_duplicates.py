# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pandas as pd

from anemoi.transform.filter import expect_tabular
from anemoi.transform.filters.tabular import TabularFilter
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


class DropDuplicates(TabularFilter, registry_name="drop_duplicates"):
    """Drop duplicate rows from a DataFrame.

    The configuration can contain either a list of column names to be
    considered or a prefix string (`column_prefix` key), where all columns
    starting with this prefix will be considered. If neither are provided, then
    all columns will be considered.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - drop_duplicates:
              columns:
                - column1
                - column2

    """

    def __init__(self, *, columns: list[str] | None = None, column_prefix: str | None = None):
        if bool(columns) and bool(column_prefix):
            raise ValueError("Either columns or column_prefix may be specified, but not both.")
        self.columns = columns
        self.column_prefix = column_prefix

    @expect_tabular
    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        if self.columns:
            subset = list(self.columns)
        elif self.column_prefix:
            subset = [col for col in obs_df.columns if col.startswith(self.column_prefix)]
            if not subset:
                raise ValueError(f"No columns starting with '{self.column_prefix}' found in DataFrame.")
        else:
            subset = None

        if subset:
            raise_if_df_missing_cols(obs_df, subset)
        obs_df = obs_df.drop_duplicates(subset=subset)

        return obs_df
