# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Transforms of selected observations and forcings of (time, grid point). Pure numpy.

Ported from obsdata (sat_view_angles.add_scan_position, compute_forcings) through obsbase,
so values match the existing gridded products.
"""

import numpy as np
import pandas as pd

from anemoi.transform.obs.config import Scan

# Planck's radiation constants for wavenumbers in m-1 (as eccodes and obsbase.bufr)
PLANCK_C1 = 1.191042972e-16  # 2 h c^2 [W m2 sr-1]
PLANCK_C2 = 1.4387768775e-2  # h c / k [m K]


def scan_position(fov: np.ndarray, scan: Scan) -> np.ndarray:
    """(k - c) / c in [-1, 1] for position k = fov // group_size, c = (n - 1) / 2: negative =
    left of the flight track (instrument frame: no flip between ascending and descending).
    """
    k = fov.astype(np.int64) // scan.group_size
    if k.size and (k.min() < 0 or k.max() >= scan.n_positions):
        raise ValueError(f"{scan.column} {fov.min()}..{fov.max()} maps outside [0, {scan.n_positions})")
    centre = (scan.n_positions - 1) / 2
    side = 1.0 if scan.first_side == "left" else -1.0
    return (side * (k - centre) / centre).astype(np.float32)


def signed_sin_vza(scan_pos: np.ndarray, zenith: np.ndarray) -> np.ndarray:
    return (np.sign(scan_pos) * np.sin(np.radians(np.abs(zenith)))).astype(np.float32)


def solar_declination(julian_day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(declination [deg], time correction [h.deg]) as earthkit-meteo's solar functions."""
    angle = julian_day / 365.25 * np.pi * 2
    declination = (
        0.396372
        - 22.91327 * np.cos(angle)
        + 4.025430 * np.sin(angle)
        - 0.387205 * np.cos(2 * angle)
        + 0.051967 * np.sin(2 * angle)
        - 0.154527 * np.cos(3 * angle)
        + 0.084798 * np.sin(3 * angle)
    )
    time_correction = (
        0.004297
        + 0.107029 * np.cos(angle)
        - 1.837877 * np.sin(angle)
        - 0.837378 * np.cos(2 * angle)
        - 2.340475 * np.sin(2 * angle)
    )
    return declination, time_correction


def cos_sza(julian_day, hours, lat, lon) -> np.ndarray:
    """Cosine of the solar zenith angle, clipped at 0 (night)."""
    declination, time_correction = solar_declination(julian_day)
    declination, lat = np.radians(declination), np.radians(lat)
    hour_angle = np.radians((hours - 12) * 15 + lon + time_correction)
    cos = np.sin(declination) * np.sin(lat) + np.cos(declination) * np.cos(lat) * np.cos(hour_angle)
    return np.clip(cos, 0, None)


def forcings(names: list[str], time: pd.Timestamp, lat: np.ndarray, lon: np.ndarray) -> dict:
    """{name: float32 array like lat} of forcings at one time and the given points."""
    julian_day = (time - time.to_period("Y").to_timestamp()) / pd.Timedelta("1D")
    hours = (time - time.normalize()) / pd.Timedelta("1h")
    local_time = (lon / 360.0 * 24.0 + hours) % 24
    year, day = julian_day / 365.25 * 2 * np.pi, local_time / 24 * 2 * np.pi
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    ones = np.ones_like(lat, dtype=np.float64)
    every = {
        "cos_latitude": lambda: np.cos(lat_r),
        "sin_latitude": lambda: np.sin(lat_r),
        "cos_longitude": lambda: np.cos(lon_r),
        "sin_longitude": lambda: np.sin(lon_r),
        "sin_julian_day": lambda: np.sin(year) * ones,
        "cos_julian_day": lambda: np.cos(year) * ones,
        "sin_local_time": lambda: np.sin(day),
        "cos_local_time": lambda: np.cos(day),
        "cos_sza": lambda: cos_sza(julian_day, hours, lat, lon),
    }
    return {n: every[n]().astype(np.float32) for n in names}


def bt_to_radiance(bt: np.ndarray, wavenumber: float) -> np.ndarray:
    """Planck radiance [W m-2 sr-1 (m-1)-1] at wavenumber [cm-1]."""
    nu = wavenumber * 100.0  # m-1
    return PLANCK_C1 * nu**3 / np.expm1(PLANCK_C2 * nu / np.asarray(bt, np.float64))


def radiance_to_bt(radiance: np.ndarray, wavenumber: float) -> np.ndarray:
    """Inverse of bt_to_radiance; radiance <= 0 or NaN gives NaN."""
    nu = wavenumber * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        bt = PLANCK_C2 * nu / np.log1p(PLANCK_C1 * nu**3 / radiance)
    return np.where(radiance > 0, bt, np.nan)
