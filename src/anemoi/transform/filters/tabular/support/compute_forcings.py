# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np


def solar_declination_angle(julian_day: np.ndarray) -> np.ndarray:
    """Vectorized version of Earthkit function:
    https://github.com/ecmwf/earthkit-meteo/blob/develop/earthkit/meteo/solar/__init__.py

    Parameters
    ----------
    julian_day
        array of julian day values

    Returns
    -------
    unknown
        array of solar declination angle values
    """
    days_per_year = 365.25
    angle = julian_day / days_per_year * np.pi * 2
    # declination in [degrees]
    declination = (
        0.396372
        - 22.91327 * np.cos(angle)
        + 4.025430 * np.sin(angle)
        - 0.387205 * np.cos(2 * angle)
        + 0.051967 * np.sin(2 * angle)
        - 0.154527 * np.cos(3 * angle)
        + 0.084798 * np.sin(3 * angle)
    )
    # time correction in [ h.degrees ]
    time_correction = (
        0.004297
        + 0.107029 * np.cos(angle)
        - 1.837877 * np.sin(angle)
        - 0.837378 * np.cos(2 * angle)
        - 2.340475 * np.sin(2 * angle)
    )
    return declination, time_correction


def cos_solar_zenith_angle(
    julian_days: np.ndarray,
    hours: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    """Calculate cosine of solar zenith angle. Vectorized version of earthkit function:
    https://github.com/ecmwf/earthkit-meteo/blob/develop/earthkit/meteo/solar/__init__.py

    Parameters
    ----------
    julian_days
        array of julian day values
    hours
        array of hours from start of jul_day
    latitudes
        array of latitude values
    longitudes
        array of longitude values

    Returns
    -------
    unknown
        array of cosine solar zenith angle values
    """
    # declination angle + time correction for solar angle
    declination, time_correction = solar_declination_angle(julian_days)
    # solar_declination_angle returns degrees
    declination = np.deg2rad(declination)
    latitudes = np.deg2rad(latitudes)
    sindec_sinlat = np.sin(declination) * np.sin(latitudes)
    cosdec_coslat = np.cos(declination) * np.cos(latitudes)
    # solar hour angle [h.deg]
    solar_angle = np.deg2rad((hours - 12) * 15 + longitudes + time_correction)
    zenith_angle = sindec_sinlat + cosdec_coslat * np.cos(solar_angle)
    # Clip negative values
    return np.clip(zenith_angle, 0, None)
