# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Literal

import pandas as pd

from anemoi.transform.filters.tabular import TabularFilter
from anemoi.transform.filters.tabular.support.sat_view_angles import calc_azimuth
from anemoi.transform.filters.tabular.support.sat_view_angles import calc_zenith
from anemoi.transform.filters.tabular.support.sat_view_angles import get_meteosat_loc


class AddMSGAngles(TabularFilter, registry_name="add_msg_angles"):
    """Adds columns representing the Meteosat angles to the DataFrame which are
    calculated from the sub-satellite latitude, longitude and observed latitude,
    longitude.

    The input DataFrame must contain a column (denoted by the configuration key
    `satellite_id`) containing the sub-satellite ID.

    The configuration for this filter can contain keys `azimuth`, `zenith` and
    `satellite_id` representing the names of these columns.

    The `angle` config key can be set to either "azimuth", "zenith" or "both"
    which determines which angle columns are added.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - add_msg_angle:
              angle: both
              azimuth: azimuth_angle
              zenith: zenith_angle
              satellite_id: satid

    """

    def __init__(
        self,
        *,
        angle: Literal["azimuth", "zenith", "both"] = "both",
        azimuth: str = "azimuth",
        zenith: str = "zenith",
        satellite_id="satellite_id",
    ):
        if angle not in ("azimuth", "zenith", "both"):
            raise ValueError(f"Invalid angle: {angle}. Must be 'azimuth', 'zenith' or 'both'.")

        self.angle = ("azimuth", "zenith") if angle == "both" else (angle,)
        self.azimuth = azimuth
        self.zenith = zenith
        self.satellite_id = satellite_id

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.satellite_id not in df.columns:
            raise ValueError(f"DataFrame must contain a column '{self.satellite_id}' for MSG angles calculation.")

        df = df.copy()
        latdeg, londeg = df["latitude"].values, df["longitude"].values
        satids, dts = df[self.satellite_id].values, df["datetime"].values
        satlats, satlons = get_meteosat_loc(satids, dts)
        if "azimuth" in self.angle:
            df[self.azimuth] = calc_azimuth(latdeg, londeg, satlats, satlons)
        if "zenith" in self.angle:
            df[self.zenith] = calc_zenith(latdeg, londeg, satlats, satlons)
        return df
