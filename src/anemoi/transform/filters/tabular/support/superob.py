# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import healpy as hp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from anemoi.transform.grids.named import lookup


def define_grid(grid: str) -> np.ndarray:
    """Return grid points as an (N, 2) array of [lat, lon] pairs."""
    data = lookup(grid)
    lat = data["latitudes"]
    lon = data["longitudes"]
    lon = np.where(lon > 180, lon - 360, lon)
    return np.column_stack([lat, lon])


def define_healpix_grid(nside: int) -> np.ndarray:
    # Calculate the number of pixels
    npix = hp.nside2npix(nside)
    # Get the theta and phi coordinates of each pixel
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    # Convert theta and phi to latitude and longitude
    lat = 90 - np.degrees(theta)  # theta=0 is North pole
    lon = np.degrees(phi)
    # Adjust longitude range from [0, 360) to [-180, 180)
    lon = np.where(lon > 180, lon - 360, lon)
    # Stack latitudes and longitudes
    grid_points = np.column_stack([lat, lon])
    return grid_points


def assign_nearest_grid(df: pd.DataFrame, grid_points: np.ndarray, time_slot_len: int) -> pd.DataFrame:
    # Make an explicit copy at the start
    df = df.copy()

    time_start = df["date"].min()
    time_end = df["date"].max()

    # Create time grid
    time_grid = pd.date_range(time_start, time_end, freq=f"{time_slot_len}s")

    # Find nearest time slot for each data point
    # Use side='right' to handle exact matches correctly
    temporal_indices = np.searchsorted(time_grid, df["date"], side="right") - 1
    # Clip to ensure no negative indices
    temporal_indices = np.clip(temporal_indices, 0, None)

    tree = cKDTree(grid_points)

    # Find nearest spatial grid points
    distances, spatial_indices = tree.query(df[["latitude", "longitude"]])

    # Assign both columns at once using assign
    return df.assign(
        grid_index=spatial_indices + len(grid_points) * temporal_indices,
        spatial_index=spatial_indices,
        distance=distances,
    )
