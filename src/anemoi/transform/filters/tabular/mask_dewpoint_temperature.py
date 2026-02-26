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

from anemoi.transform.filter import Filter
from anemoi.transform.filters.tabular import filter_registry
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


@filter_registry.register("mask_dewpoint_temperature")
class MaskDewpointTemperature(Filter):
    """Mask the dewpoint temperature column (and optionally the specific
    humidity column) of a DataFrame if the temperature is less than the dewpoint
    temperature.

    The configuration should contain the column names of the temperature,
    specific humidity and dewpoint temperature columns.
    The `mask_specific_humidity` (boolean) key can be used to mask the specific
    humidity column using the same mask (default false).

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - mask_td:
              mask_specific_humidity: true
              temperature: 2t
              dewpoint_temperature: 2td
              specific_humidity: 2q

    """

    def __init__(
        self,
        *,
        temperature: str = "2t",
        dewpoint_temperature: str = "2d",
        specific_humidity: str = "2q",
        mask_specific_humidity: bool = False,
    ):
        self.temperature = temperature
        self.dewpoint_temperature = dewpoint_temperature
        self.specific_humidity = specific_humidity
        self.mask_specific_humidity = mask_specific_humidity

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        required_cols = [self.temperature, self.dewpoint_temperature]
        if self.mask_specific_humidity:
            required_cols.append(self.specific_humidity)
        raise_if_df_missing_cols(obs_df, required_cols)

        logging.info(
            "Masking dewpoint temperature column if the temperature column is less than the dewpoint temperature column"
        )
        mask = obs_df[self.temperature] < obs_df[self.dewpoint_temperature]
        obs_df[self.dewpoint_temperature] = obs_df[self.dewpoint_temperature].mask(mask)
        if self.mask_specific_humidity:
            obs_df[self.specific_humidity] = obs_df[self.specific_humidity].mask(mask)
        return obs_df
