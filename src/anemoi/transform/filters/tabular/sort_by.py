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

from anemoi.transform.filters.tabular import TabularFilter
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


class SortBy(TabularFilter, registry_name="sort_by"):
    """Sort a DataFrame by the list of columns supplied in the config.

    The configuration should contain a list of column names to sort by under
    the 'columns' key.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - sort_by:
              columns:
                - foo
                - bar

    """

    def __init__(self, *, columns: list[str]):
        self.columns = columns

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        raise_if_df_missing_cols(df, self.columns)
        df_sorted = df.copy()

        logging.info(f"Sorting by columns: {self.columns}")

        try:
            df_sorted = df_sorted.sort_values(by=self.columns, kind="stable")
        except Exception as e:
            logging.error(f"Error sorting DataFrame: {type(e).__name__}: {str(e)}")
            for col in self.columns:
                try:
                    _ = df_sorted[col]
                    logging.info(f"Successfully accessed column: {col}")
                except Exception as e:
                    logging.error(f"Failed to access column {col}: {str(e)}")
                    raise
            return df_sorted

        return df_sorted
