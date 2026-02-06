# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from datetime import datetime
from typing import Tuple

import numpy as np


def calc_zenith(latdeg, londeg, satlats, satlons):
    # Assume geostationary orbit here
    rearth = 6378.170
    satalt = 6610839 * 1.0e-6 * rearth - rearth

    # Conversion to radians
    rsatlons = np.radians(satlons)
    rsatlats = np.radians(satlats)
    rlons = np.radians(londeg)
    rlats = np.radians(latdeg)

    # Distance on the ground
    ds = (
        rearth
        * 2
        * np.arcsin(
            np.sqrt(
                (1 - np.sin(rlats) * np.sin(rsatlats) - np.cos(rlats) * np.cos(rsatlats) * np.cos(rlons - rsatlons)) / 2
            )
        )
    )

    # Trigonometry
    a = ds / rearth
    rl = rearth * np.sin(a)
    rm = rearth * np.cos(a)
    tb = rl / (rearth + satalt - rm)
    b = np.arctan(tb)
    z = a + b

    # Convert to degrees
    zenith = np.degrees(z)
    return zenith


def calc_azimuth(latdeg, londeg, satlats, satlons):

    # Initialize azimuth array
    azm = np.zeros_like(latdeg)

    # Create mask for valid calculations
    mask = (np.abs(latdeg - satlats) > 0.00001) & (np.abs(londeg - satlons) > 0.00001)

    # Convert degrees to radians for masked elements
    lat = np.radians(latdeg[mask])
    lon = np.radians(londeg[mask])
    latS = np.radians(satlats[mask])
    lonS = np.radians(satlons[mask])

    # Calculate angular distance
    zdlon = lonS - lon
    zdlat = latS - lat
    za = np.sin(zdlat / 2.0) ** 2 + np.cos(lat) * np.cos(latS) * np.sin(zdlon / 2.0) ** 2
    distOS = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(za)))

    # Calculate azimuth using law of sines
    azmsin = np.cos(latS) / np.sin(distOS) * np.sin(lon - lonS)
    azmsin = np.clip(azmsin, -1.0, 1.0)
    azmsin = np.arcsin(azmsin)

    # Calculate azimuth using law of cosines
    azmcos = (np.sin(latS) - np.sin(lat) * np.cos(distOS)) / (np.cos(lat) * np.sin(distOS))
    azmcos = np.clip(azmcos, -1.0, 1.0)
    azmcos = np.arccos(azmcos)

    # Adjust azmcos based on azmsin
    azmcos = np.where(azmsin > 0, -azmcos, azmcos)

    # Final azimuth calculation
    azm_calc = azmcos
    azm_calc = np.where(azm_calc >= np.pi, azm_calc - 2.0 * np.pi, azm_calc)
    azm_calc = np.where(azm_calc < -np.pi, azm_calc + 2.0 * np.pi, azm_calc)
    azm_calc = np.degrees(azm_calc)

    # Ensure azimuth angle is in range 0-360
    azm_calc = np.mod(azm_calc, 360.0)

    # Assign calculated values to the output array
    azm[mask] = azm_calc

    return azm


def get_meteosat_loc(satids: np.ndarray, dts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given an array of satellite IDs and an arrays of datetimes,
        returns the sub-satellite lat, lon for that satellite.
        Adapted from obd/bufr2odb/b2o_convert_asr.F90

    Parameters
    ----------
    satids
        array of WMO satellite IDs (one per pixel)
    dts
        array of datetimes for each pixel

    Returns
    -------
    unknown
        arrays of sub-satellite latitudes and longitudes in degrees
    """
    lons = np.zeros_like(satids)
    lats = np.zeros_like(satids)

    mask = (satids == 55) & (dts > np.datetime64(datetime.strptime("20161020", "%Y%m%d")))
    lons[mask] = 41.5

    mask = (satids == 56) & (dts > np.datetime64(datetime.strptime("20220508", "%Y%m%d")))
    lons[mask] = 45.5

    mask = (satids == 57) & (dts < np.datetime64(datetime.strptime("20130124", "%Y%m%d")))
    lons[mask] = -3.4

    mask = (satids == 70) & (dts < np.datetime64(datetime.strptime("20151201", "%Y%m%d")))
    lons[mask] = -3.4

    return lats, lons
