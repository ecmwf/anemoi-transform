# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pandas as pd

from anemoi.transform.filters.tabular import TabularFilter
from anemoi.transform.filters.tabular import filter_registry
from anemoi.transform.filters.tabular.support.compute_forcings import cos_solar_zenith_angle


@filter_registry.register("add_forcings")
class AddForcings(TabularFilter):
    """Adds forcings columns to the DataFrame.

    The configuration should be a dictionary with the key 'columns' containing
    a list of the requested forcings, which must be selected from:
      - cos_julian_day
      - sin_julian_day
      - cos_sza
      - sin_local_time
      - cos_local_time
      - cos_latitude
      - sin_latitude
      - cos_longitude
      - sin_longitude

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - add_forcings:
              columns:
                - cos_julian_day
                - sin_julian_day

    """

    SUPPORTED_FORCINGS = {
        "cos_julian_day",
        "sin_julian_day",
        "cos_sza",
        "sin_local_time",
        "cos_local_time",
        "cos_latitude",
        "sin_latitude",
        "cos_longitude",
        "sin_longitude",
    }

    def __init__(self, *, columns: list[str]):
        if not set(columns).issubset(self.SUPPORTED_FORCINGS):
            raise ValueError(f"Unknown columns requested: {set(columns) - self.SUPPORTED_FORCINGS}")
        self.columns = columns

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        # Make a copy to avoid modifying the input DataFrame
        obs_df = obs_df.copy()

        date = pd.DatetimeIndex(obs_df["datetime"].values)
        longitude = obs_df["longitude"].values
        latitude = obs_df["latitude"].values
        for column in self.columns:
            if column == "cos_sza":
                obs_df.loc[:, column] = self._cos_solar_zenith_angle(date, latitude, longitude)
                continue
            trig_function, variable = column.split("_", maxsplit=1)
            angle_radians = self._angle_radians(variable, date, latitude, longitude)

            func = getattr(np, trig_function)
            obs_df.loc[:, column] = func(angle_radians)
        return obs_df

    @staticmethod
    def _angle_radians(name: str, date: pd.DatetimeIndex, latitude: np.ndarray, longitude: np.ndarray):
        if name == "julian_day":
            return AddForcings._julian_day_radians(date)
        elif name == "local_time":
            return AddForcings._local_time_radians(date, longitude)
        elif name == "latitude":
            return np.deg2rad(latitude)
        elif name == "longitude":
            return np.deg2rad(longitude)
        raise ValueError(f"Unknown angle name: {name}")

    @staticmethod
    def _julian_day(date):
        delta = date - date.to_period("Y").to_timestamp()
        julian_day = delta.days + delta.seconds / 86400.0
        return julian_day

    @staticmethod
    def _julian_day_radians(date):
        julian_day = AddForcings._julian_day(date)
        radians = julian_day / 365.25 * np.pi * 2
        return radians

    @staticmethod
    def _hours_since_midnight(date):
        delta = date - date.to_period("D").to_timestamp()
        hours_since_midnight = (delta.days + delta.seconds / 86400.0) * 24
        return hours_since_midnight

    @staticmethod
    def _cos_solar_zenith_angle(date, latitude, longitude):
        julian_day = AddForcings._julian_day(date)
        hours_since_midnight = AddForcings._hours_since_midnight(date)
        return cos_solar_zenith_angle(julian_day, hours_since_midnight, latitude, longitude)

    @staticmethod
    def _local_time_radians(date, longitude):
        hours_since_midnight = AddForcings._hours_since_midnight(date)
        local_time = (longitude / 360.0 * 24.0 + hours_since_midnight) % 24
        radians = local_time / 24 * np.pi * 2
        return radians
