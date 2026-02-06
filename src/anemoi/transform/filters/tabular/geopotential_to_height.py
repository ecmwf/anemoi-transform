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


class GeopotentialToHeight(TabularFilter, registry_name="geopotential_to_height"):
    """Converts geopotential height to height.

    The `geopotential` config key defines the name of the column containing
    geopotential height (must exist in the DataFrame). The `height` config key
    defines the name of the column containing height following conversion
    (i.e. the result) - if not passed in (or None), this will overwrite the
    geopotential column.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - geopotential_to_height:
              geopotential: z
              height: z

    """

    def __init__(self, *, geopotential, height=None):
        self.geopotential = geopotential
        self.height = height if height else geopotential

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        raise_if_df_missing_cols(obs_df, [self.geopotential])
        logging.info("Converting height to geopotential")
        obs_df[self.height] = obs_df[self.geopotential].apply(lambda x: x / 9.80665)
        return obs_df
