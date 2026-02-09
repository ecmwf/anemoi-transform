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
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


class Clip(TabularFilter, registry_name="clip"):
    """Clips columns of a DataFrame to the specified range.

    The configuration should be a dictionary of column names and clip ranges,
    where a clip range is a list containing the minimum and maximum values
    (values of null can be used to indicate no minimum or maximum).

    The resulting column will have the same name as the original column (i.e.
    will not be renamed).

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - clip:
              precipitation: [0, 100]

    """

    def __init__(self, **config):
        if not config:
            raise ValueError("No columns to clip were specified.")
        for column, clip_range in config.items():
            if not isinstance(clip_range, (list, tuple)) or len(clip_range) != 2:
                raise ValueError(f"Invalid clip range for column {column}: {clip_range}")
            if not all(isinstance(v, (int, float)) or v is None for v in clip_range):
                raise ValueError(f"Clip range values for column {column} must be numeric or None: {clip_range}")
        self.config = config

    @expect_tabular
    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        raise_if_df_missing_cols(obs_df, self.config.keys())

        for col, clip_range in self.config.items():
            logging.info(f"Clipping {col} to {clip_range}")
            obs_df.loc[:, col] = obs_df[col].clip(*clip_range)
        return obs_df
