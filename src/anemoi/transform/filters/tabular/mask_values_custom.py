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
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


@filter_registry.register("mask_values_custom")
class MaskValuesCustom(Filter):
    """Mask values in specific columns based on custom conditions that can
    reference multiple columns.

    Unlike the ``mask`` filter, this filter allows you to mask a column based on
    conditions involving other columns using pandas query-style expressions.

    The configuration is a dictionary mapping column names to query expressions.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - mask_values_custom:
              fg_depar_z_0: "abs(fg_depar_z_0) > 500 & pressure < 50000"

    """

    def __init__(self, **config):
        if not config:
            raise ValueError("No columns to mask were specified.")
        self.config = config

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        raise_if_df_missing_cols(obs_df, self.config.keys())
        for col, condition_str in self.config.items():
            logging.info(f"Masking {col} with custom condition: {condition_str}")

            try:
                # Evaluate the condition as a pandas query to get boolean mask
                # The condition can reference any column in the dataframe
                mask = obs_df.eval(condition_str)

                # Mask values where condition is True
                obs_df[col] = obs_df[col].mask(mask)

                n_masked = mask.sum()
                logging.info(f"Masked {n_masked} values in {col} ({n_masked/len(obs_df)*100:.2f}%)")

            except Exception as e:
                logging.error(f"Error evaluating custom mask condition for {col}: {e}")
                raise ValueError(f"Invalid condition for column '{col}': {condition_str}. Error: {e}")

        return obs_df
