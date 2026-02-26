# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pandas as pd

from anemoi.transform.filter import Filter
from anemoi.transform.filters.tabular import filter_registry
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


@filter_registry.register("rename")
class Rename(Filter):
    """Rename one or more columns in the DataFrame.

    The configuration should be a dictionary with the key 'columns' containing a
    dictionary which maps old column names to new column names.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - rename:
              columns:
                from_name: to_name

    """

    def __init__(self, *, columns: dict[str, str]):
        self.columns = columns

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        raise_if_df_missing_cols(obs_df, list(self.columns.keys()))
        obs_df = obs_df.rename(columns=self.columns)
        return obs_df
