# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pandas as pd

from anemoi.transform.filters.tabular import TabularFilter
from anemoi.transform.filters.tabular.support.sat_view_angles import calc_azimuth
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


class AddAzimuth(TabularFilter, registry_name="add_azimuth"):
    """Adds a column to the DataFrame containing the viewing azimuth angle
    calculated from the latitude, longitude and the spacecraft latitude,
    spacecraft longitude.

    The `azimuth`, `spacecraft_latitude` and `spacecraft_longitude` config keys
    define the names of the associated columns in the input DataFrame.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - add_azimuth:
              azimuth: azimuth_angle
              spacecraft_latitude: sc_lat
              spacecract_longitude: sc_lon

    """

    def __init__(
        self,
        *,
        azimuth: str = "azimuth",
        spacecraft_latitude: str = "spacecraft_latitude",
        spacecraft_longitude: str = "spacecraft_longitude",
    ):
        self.azimuth = azimuth
        self.spacecraft_latitude = spacecraft_latitude
        self.spacecraft_longitude = spacecraft_longitude

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        raise_if_df_missing_cols(obs_df, required_cols=[self.spacecraft_latitude, self.spacecraft_longitude])

        lats, lons = obs_df["latitude"].values, obs_df["longitude"].values
        sc_lats, sc_lons = obs_df[self.spacecraft_latitude].values, obs_df[self.spacecraft_longitude].values

        obs_df[self.azimuth] = calc_azimuth(lats, lons, sc_lats, sc_lons)

        return obs_df
