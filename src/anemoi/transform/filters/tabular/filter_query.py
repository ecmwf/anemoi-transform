# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction

import logging

import pandas as pd

from anemoi.transform.filter import Filter
from anemoi.transform.filters.tabular import filter_registry


@filter_registry.register("filter_query")
class FilterQuery(Filter):
    """Filter rows of a DataFrame using a SQL-like query expression.

    This filter uses ``pandas.DataFrame.query()`` to filter rows based on a
    single query string that can reference multiple columns, similar to SQL
    ``WHERE`` clauses.

    Notes
    -----
      - Uses pandas ``query()`` syntax
      - Supports: ``in``, ``and``, ``or``, ``not``, ``<``, ``>``, ``<=``, ``>=``, ``==``, ``!=``
      - Can use functions: ``abs()``, etc.
      - Rows matching the condition are KEPT, others are dropped

    Examples
    --------
    Filter by multiple conditions:

    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - filter_query:
              query: "varno in [1,2,3,4,7,59] and pressure in [85000, 92500, 100000]"

    Filter by ranges and comparisons:

    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - filter_query:
              query: "abs(fg_depar_t_0) < 15 and pressure > 50000 and abs(latitude) < 80"

    Complex logical conditions:

    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - filter_query:
              query: "(varno == 2 and abs(fg_depar_t_0) < 15) or (varno == 7 and abs(fg_depar_q_0) < 0.005)"

    """

    def __init__(self, *, query: str):
        if not query:
            raise ValueError("Query expression cannot be empty")

        self.query = query

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Filtering rows with query: {self.query}")
        initial_len = len(obs_df)

        try:
            # Filter using pandas query
            obs_df = obs_df.query(self.query)

            final_len = len(obs_df)
            n_dropped = initial_len - final_len
            logging.info(f"Dropped {n_dropped} rows ({n_dropped/initial_len*100:.2f}%). Kept {final_len} rows.")

        except Exception as e:
            logging.error(f"Error evaluating query filter: {e}")
            raise ValueError(f"Invalid query expression: {self.query}. Error: {e}")

        return obs_df
