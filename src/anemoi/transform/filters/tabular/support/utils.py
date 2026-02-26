# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def raise_if_df_missing_cols(df: pd.DataFrame, required_cols: Iterable[str]):
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame is missing columns: {missing_cols}. Available columns: {df.columns}")


def get_heights(
    heights_lats: np.ndarray,
    heights_lons: np.ndarray,
    heights: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """Given 2D arrays of lat, lons and heights finds the nearest heights for points in lat, lon arrays

    Parameters
    ----------
    heights_lats
        array of orog field lats
    heights_lons
        array of orog field lons
    heights
        array of orog field heights
    lat
        array of latitude points
    lon
        array of longitude points

    Returns
    -------
    unknown
        array of heights closest to lat lon points
    """
    lat_tree = cKDTree(np.c_[heights_lats])
    lon_tree = cKDTree(np.c_[heights_lons])
    _, lat_idxs = lat_tree.query(lat.reshape(-1, 1))
    _, lon_idxs = lon_tree.query(lon.reshape(-1, 1))
    return heights[(lat_idxs, lon_idxs)]
