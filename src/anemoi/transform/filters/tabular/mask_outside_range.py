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

from anemoi.transform.filter import expect_tabular
from anemoi.transform.filters.tabular import TabularFilter


class MaskOutsideRange(TabularFilter, registry_name="mask_outside_range"):
    """Mask values outside of specified ranges in a DataFrame.

    The configuration should be a dictionary of column names and ranges, where a
    range is a tuple containing the minimum and maximum values (values of nil
    can be used to indicate no lower/upper bound).

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - mask_outside_range:
              my_column: [1.0, 2.0]

    """

    def __init__(self, **config):
        if not config:
            raise ValueError("No columns to mask were specified.")

        for column, mask_range in config.items():
            if not isinstance(mask_range, (list, tuple)) or len(mask_range) != 2:
                raise ValueError(f"Invalid mask range for column {column}: {mask_range}")
            if not all(isinstance(v, (int, float)) or v is None for v in mask_range):
                raise ValueError(f"Mask range values for column {column} must be numeric or None: {mask_range}")

        self.config = config

    @expect_tabular
    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        for column, range in self.config.items():
            min_val, max_val = range
            logging.info(f"Masking {column} with condition: {min_val} <= {column} <= {max_val}")
            obs_df[column] = obs_df[column].mask(
                (obs_df[column] < min_val if min_val is not None else False)
                | (obs_df[column] > max_val if max_val is not None else False)
            )
        return obs_df
