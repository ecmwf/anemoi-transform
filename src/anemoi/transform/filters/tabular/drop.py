# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pandas as pd

from anemoi.transform.filters.tabular import TabularFilter
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


class Drop(TabularFilter, registry_name="drop"):
    """Drop one or more columns from a DataFrame.

    The configuration should be a dictionary with the key 'columns' containing
    a list of column names to drop.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - drop:
              columns:
                - column1
                - column2

    """

    def __init__(self, *, columns: list[str]):
        if not columns:
            raise ValueError("No columns to drop were specified.")
        self.columns = columns

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        raise_if_df_missing_cols(obs_df, self.columns)
        obs_df = obs_df.drop(labels=self.columns, axis=1)
        return obs_df
