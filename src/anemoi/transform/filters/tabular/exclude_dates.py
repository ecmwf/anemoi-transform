# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections import defaultdict

import pandas as pd

from anemoi.transform.filter import expect_tabular
from anemoi.transform.filters.tabular import TabularFilter
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


class ExcludeDates(TabularFilter, registry_name="exclude_dates"):
    """Masks values in specified columns of a DataFrame if the 'datetime' column
    falls within any of the date ranges provided (inclusive).

    Keys of the config dictionary are column names, with values which are lists
    of date ranges (list of two dates).

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - exclude_dates:
                obsvalue_rawbt_4:
                  - [19980505, 20010101]
                  - [20210101, 20230501]
                obsvalue_rawbt_6:
                  - [20040505, 20070101]
                  - [20200101, 20210101]

    """

    def __init__(self, **config):
        if not config:
            raise ValueError("No columns to exclude dates from were specified.")

        excluded_dates = defaultdict(list)
        for column, ranges in config.items():
            if not (ranges and isinstance(ranges, (list, tuple))):
                raise ValueError(f"Invalid date ranges {ranges} for column '{column}'. Expected a list/tuple.")

            if len(ranges) == 2 and all(isinstance(range, (int, str)) for range in ranges):
                # normalize ranges to a list of tuples if a single tuple was provided.
                ranges = [ranges]

            for date_range in ranges:
                try:
                    start, end = date_range
                except ValueError as e:
                    raise ValueError(
                        f"Invalid date range {date_range} for column '{column}'. "
                        "Expected a tuple like (start_date, end_date)."
                    ) from e

                start_dt = pd.to_datetime(str(start), format="%Y%m%d")
                end_dt = pd.to_datetime(str(end), format="%Y%m%d")
                end_dt = end_dt + pd.Timedelta(days=1)  # So that it masks all data on the specified end date

                date_range = (start_dt, end_dt)
                excluded_dates[column].append(date_range)

        self.excluded_dates = excluded_dates

    @expect_tabular
    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        required_cols = list(self.excluded_dates.keys()) + ["datetime"]
        raise_if_df_missing_cols(obs_df, required_cols)

        for column, date_ranges in self.excluded_dates.items():
            for start_dt, end_dt in date_ranges:
                logging.info(f"Excluding dates for column '{column}' when datetime is between {start_dt} and {end_dt}.")

                # Create a mask based on the 'datetime' column.
                mask = (obs_df["datetime"] >= start_dt) & (obs_df["datetime"] < end_dt)

                # Mask the specified column where the condition is True.
                obs_df[column] = obs_df[column].mask(mask)
        return obs_df
