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

from anemoi.transform.filter import expect_tabular
from anemoi.transform.filters.tabular import TabularFilter
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


class MaskInfs(TabularFilter, registry_name="mask_infs"):
    """Mask values in a DataFrame where columns (defined via configuration)
    contain infinite values.

    The configuration can contain either a list of column names to be
    considered or a prefix string (`column_prefix` key), where all columns
    starting with this prefix will be considered.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - mask_infs:
              columns:
                - foo
                - bar

    """

    def __init__(self, *, columns: list[str] | None = None, column_prefix: str | None = None):
        if bool(columns) == bool(column_prefix):
            raise ValueError("Either columns or column_prefix must be specified, but not both.")
        self.columns = columns
        self.column_prefix = column_prefix

    @expect_tabular
    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        if self.columns:
            columns = list(self.columns)
            raise_if_df_missing_cols(obs_df, self.columns)
        elif self.column_prefix:
            columns = [col for col in obs_df.columns if col.startswith(self.column_prefix)]
            if not columns:
                raise ValueError(f"No columns starting with '{self.column_prefix}' found in DataFrame.")

        for col in columns:
            logging.info(f"Masking {col} with infinite values")
            obs_df[col] = obs_df[col].mask(obs_df[col] == np.inf)
            obs_df[col] = obs_df[col].mask(obs_df[col] == -np.inf)
        return obs_df
