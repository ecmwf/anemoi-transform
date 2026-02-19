import logging

import numpy as np
import pandas as pd

from anemoi.transform.filters.tabular import TabularFilter
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


class MaskValues(TabularFilter, registry_name="mask"):
    """Mask the columns of a DataFrame based on a condition.

    The configuration should be a dictionary with column names as keys and
    the mask condition (a string) as a value. This string can contain "inf" to
    represent infinity.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - mask:
              foo: "lambda x: x >= 2"

    """

    def __init__(self, **config):
        if not config:
            raise ValueError("No columns to mask were specified.")
        self.config = config

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        raise_if_df_missing_cols(obs_df, self.config.keys())

        for col, mask_condition in self.config.items():
            logging.info(f"Masking {col} with condition: {mask_condition}")
            namespace = {"inf": np.inf}
            mask_condition = eval(mask_condition, namespace)
            obs_df[col] = obs_df[col].mask(mask_condition)
        return obs_df
