# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Grid points and nearest-cell lookup on the unit sphere (exact at the dateline and poles).

Octahedral "oNNN" grids are built here (row sizes 20 + 4i from the pole, Gaussian
latitudes), north to south and each row east from 0°, as in MARS/GRIB; any other name comes
from anemoi.transform.grids.named.
"""

from functools import cache

import numpy as np
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0


@cache
def grid_points(grid: str) -> tuple[np.ndarray, np.ndarray]:
    """(lat, lon) in degrees of every grid point, lon in [0, 360)."""
    if grid[0].lower() == "o" and grid[1:].isdigit():
        half = 20 + 4 * np.arange(int(grid[1:]))
        sizes = np.concatenate([half, half[::-1]])
        x, _ = np.polynomial.legendre.leggauss(len(sizes))
        row_lats = np.degrees(np.arcsin(x))[::-1]  # leggauss returns south to north
        lat = np.repeat(row_lats, sizes)
        lon = np.concatenate([np.arange(n) * 360.0 / n for n in sizes])
        return lat, lon
    from anemoi.transform.grids.named import lookup

    points = lookup(grid)
    return np.asarray(points["latitudes"], np.float64), np.asarray(points["longitudes"], np.float64) % 360


def mean_spacing_km(grid: str) -> float:
    """Side of the square with a cell's mean area: o96 ~112.5 km."""
    return EARTH_RADIUS_KM * float(np.sqrt(4 * np.pi / grid_points(grid)[0].size))


def to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


@cache
def _tree(grid: str) -> cKDTree:
    return cKDTree(to_xyz(*grid_points(grid)))


def cell_index(grid: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Index of the great-circle nearest grid point (exact, via a KD-tree on the unit sphere)."""
    _, idx = _tree(grid).query(to_xyz(lat, lon))
    return idx.astype(np.int32)


def cell_offsets_km(
    grid: str, lat: np.ndarray, lon: np.ndarray, cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(distance, east, north) [km] of each obs from its cell centre: great-circle distance,
    and the obs projected on the centre's tangent plane (no wrap at the dateline or poles).
    """
    glat, glon = (np.radians(g[cell]) for g in grid_points(grid))
    p = to_xyz(lat, lon)
    centre = np.column_stack([np.cos(glat) * np.cos(glon), np.cos(glat) * np.sin(glon), np.sin(glat)])
    east = -np.sin(glon) * p[:, 0] + np.cos(glon) * p[:, 1]
    north = -np.sin(glat) * np.cos(glon) * p[:, 0] - np.sin(glat) * np.sin(glon) * p[:, 1] + np.cos(glat) * p[:, 2]
    cos = np.einsum("ij,ij->i", p, centre)
    return (
        EARTH_RADIUS_KM * np.arccos(np.clip(cos, -1, 1)),
        EARTH_RADIUS_KM * east,
        EARTH_RADIUS_KM * north,
    )
