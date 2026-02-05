# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np
import pandas as pd
import zarr

from anemoi.transform.filters.tabular import TabularFilter
from anemoi.transform.filters.tabular.support.utils import get_heights
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


class FillHeights(TabularFilter, registry_name="fill_orography"):
    """Fills missing values in the station altitude column by matching indices
    of missing station values with heights from a high-resolution orography
    file.

    The `station_altitude` (default 'stalt') config key defines the column name
    of the station altitudes in the DataFrame. The `orography_file` config key
    defines the path to the orography file. The `orography_altitude` (default
    'z'), `orography_latitude` (default 'latitude') and `orography_longitude`
    (default 'longitude') config keys define the names of the associated columns
    in the orography file.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - fill_orography:
              station_altitude: stalt
              orography_file: /path/to/orography.zarr
              orography_altitude: z
              orography_latitude: latitude
              orography_longitude: longitude

    """

    def __init__(
        self,
        *,
        orography_file: str,
        station_altitude: str = "stalt",
        orography_altitude: str = "z",
        orography_latitude: str = "latitude",
        orography_longitude: str = "longitude",
    ):
        self.orography_file = orography_file
        self.station_altitude = station_altitude
        self.orography_altitude = orography_altitude
        self.orography_latitude = orography_latitude
        self.orography_longitude = orography_longitude

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        raise_if_df_missing_cols(obs_df, [self.station_altitude])
        stalt_arr = obs_df[self.station_altitude].values
        lats = obs_df["latitude"].values
        lons = obs_df["longitude"].values

        ds_orog = zarr.open(self.orography_file, mode="r")
        lats_orog = np.array(ds_orog[self.orography_latitude])
        lons_orog = np.array(ds_orog[self.orography_longitude])
        heights = np.array(ds_orog[self.orography_altitude])

        logging.info("Finding NaN station heights...")
        nan_idxs = np.argwhere(((np.isnan(stalt_arr)) | (stalt_arr == 9999.0))).flatten()
        logging.info(f"Found {len(nan_idxs)} NaN values in station altitudes!")
        logging.info("Finding closest altitudes from orography file...")
        fill_heights = get_heights(
            lats_orog,
            lons_orog,
            heights,
            lats[nan_idxs],
            lons[nan_idxs],
        )
        obs_df[self.station_altitude].values[nan_idxs] = fill_heights
        return obs_df
